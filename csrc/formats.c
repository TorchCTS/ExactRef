/* The rules of exactref's element formats (internal): what a floating-point format's definition
   implies (its exponent range and special encodings), decoding an encoding to its value, and
   rounding a value to an encoding in the five IEEE rounding directions with each overflow mode;
   and the same for integer formats. formats.py holds the definitions (field widths, the scheme of
   special values, the bias, given NaN encodings); their rules are here only. Encodings are
   natural numbers of any width, since a format may be wider than 64 bits. */
#include "exactref.h"

/* Where a dropped remainder lies, against half the kept quantum. */
#define REST_ZERO 0
#define REST_LOW 1
#define REST_HALF 2
#define REST_HIGH 3

/* Sets f's derived fields from its definition (the default bias when bias is NULL). */
int floatfmt_set(floatfmt *f, int64_t exp_bits, int64_t man_bits, int special, const int64_t *bias, int whole_bytes) {
    if (exp_bits < 2 || man_bits < 1) return EXACTREF_FORMAT_BITS;
    if (whole_bytes && (1 + exp_bits % 8 + man_bits % 8) % 8) return EXACTREF_FORMAT_BYTES;
    if (whole_bytes && 1 + exp_bits + man_bits > 64) return EXACTREF_FORMAT_SIZE;
    /* such a format's binades span more than 2**21, whatever its bias */
    if (exp_bits > 40 || man_bits > ((int64_t)1 << 40)) return EXACTREF_FORMAT_RANGE;
    int64_t all_ones = ((int64_t)1 << exp_bits) - 1;
    f->exp_bits = exp_bits;
    f->man_bits = man_bits;
    f->special = special;
    f->bias = bias ? *bias : ((int64_t)1 << (exp_bits - 1)) - (special == SPECIAL_FNUZ ? 0 : 1);
    if (f->bias > ((int64_t)1 << 62) || f->bias < -((int64_t)1 << 62)) return EXACTREF_FORMAT_RANGE;
    /* the largest finite value: below the all-ones exponent in IEEE formats; in fn formats, below
       the all-ones exponent and mantissa (NaN) */
    f->max_exp_field = special == SPECIAL_IEEE ? all_ones - 1 : all_ones;
    f->emin = 1 - f->bias;
    f->emax = f->max_exp_field - f->bias;
    f->nan[0] = f->nan[1] = NULL;
    if (f->emax + 1 > EXACT_LIMIT || f->emin - man_bits < 1 - EXACT_LIMIT) return EXACTREF_FORMAT_RANGE;
    return 0;
}

/* f = g with g's NaN encodings, when its definition gives them, copied into own (initialized; the
   caller frees it), which f then points to: a format kept past the encodings g points to. */
int floatfmt_copy(floatfmt *f, bigint own[2], const floatfmt *g) {
    *f = *g;
    for (int s = 0; s < 2; s++) {
        if (!g->nan[s]) continue;
        if (big_copy(&own[s], g->nan[s])) return EXACTREF_NOMEM;
        f->nan[s] = &own[s];
    }
    return 0;
}

/* bits = sign * 2**(e + m) + exp_field * 2**m + man, for man < 2**m (man may be NULL for 0). */
static int assemble(bigint *bits, const floatfmt *f, int neg, int64_t exp_field, const bigint *man) {
    int rc = big_set_word(bits, (uint64_t)(neg != 0) << f->exp_bits | (uint64_t)exp_field, 0);
    if (!rc) rc = big_shl(bits, bits, f->man_bits);
    if (!rc && man) rc = big_add(bits, bits, man);
    return rc;
}

/* The encoding of zero, an infinity, NaN or the largest finite value with the given sign. */
int floatfmt_special(bigint *bits, const floatfmt *f, int which, int neg) {
    int64_t all_ones = ((int64_t)1 << f->exp_bits) - 1, m = f->man_bits;
    bigint man;
    big_init(&man);
    int rc = 0;
    switch (which) {
    case ENCODING_ZERO:
        rc = assemble(bits, f, neg && f->special != SPECIAL_FNUZ, 0, NULL);
        break;
    case ENCODING_INFINITY:
        rc = assemble(bits, f, neg, all_ones, NULL);
        break;
    case ENCODING_NAN:
        if (f->nan[neg != 0]) {
            rc = big_copy(bits, f->nan[neg != 0]);
        } else if (f->special == SPECIAL_FNUZ) {
            rc = assemble(bits, f, 1, 0, NULL); /* the negative-zero encoding */
        } else if (f->special == SPECIAL_FN) {
            if (!(rc = big_set_word(&man, 1, 0)) && !(rc = big_shl(&man, &man, m)) && !(rc = big_set_si(bits, -1)))
                rc = big_add(&man, &man, bits);
            if (!rc) rc = assemble(bits, f, neg, all_ones, &man);
        } else {
            /* the quiet NaN */
            if (!(rc = big_set_word(&man, 1, 0))) rc = big_shl(&man, &man, m - 1);
            if (!rc) rc = assemble(bits, f, neg, all_ones, &man);
        }
        break;
    default:
        /* every mantissa bit set, or all but the last in an fn format */
        if (!(rc = big_set_word(&man, 1, 0)) && !(rc = big_shl(&man, &man, m)) &&
            !(rc = big_set_si(bits, f->special == SPECIAL_FN ? -2 : -1)))
            rc = big_add(&man, &man, bits);
        if (!rc) rc = assemble(bits, f, neg, f->max_exp_field, &man);
    }
    big_free(&man);
    return rc;
}

/* The kind of an encoding from its fields, and for a finite value its exponent and whether its
   significand has the implicit bit: the rule of every decoding. */
static void classify(int *kind, int *neg, int64_t *exp, int *implicit, const floatfmt *f, int64_t exp_field, int man_zero, int man_ones) {
    int64_t all_ones = ((int64_t)1 << f->exp_bits) - 1;
    *kind = VALUE_FINITE;
    *exp = 0;
    *implicit = 0;
    if (exp_field == all_ones && (f->special == SPECIAL_IEEE || (f->special == SPECIAL_FN && man_ones))) {
        *kind = f->special == SPECIAL_IEEE && man_zero ? VALUE_INFINITE : VALUE_NAN;
    } else if (f->special == SPECIAL_FNUZ && *neg && !exp_field && man_zero) {
        *kind = VALUE_NAN; /* the negative-zero encoding: the only NaN, with no sign */
        *neg = 0;
    } else if (exp_field || !man_zero) {
        *implicit = exp_field != 0;
        *exp = (exp_field ? exp_field - f->bias : f->emin) - f->man_bits;
    } /* else a zero: exponent 0, as every zero dyadic is 0 2**0 (exactref.h) */
}

/* floatfmt_decode for a format of at most 64 bits, on a word. */
void floatfmt_decode_word(int *kind, int *neg, uint64_t *sig, int64_t *exp, const floatfmt *f, uint64_t bits) {
    int64_t m = f->man_bits;
    uint64_t mask = ((uint64_t)1 << m) - 1, man = bits & mask;
    int64_t exp_field = (int64_t)(bits >> m & (((uint64_t)1 << f->exp_bits) - 1));
    int implicit;
    *neg = (int)(bits >> (f->exp_bits + m) & 1);
    classify(kind, neg, exp, &implicit, f, exp_field, !man, man == mask);
    *sig = *kind != VALUE_FINITE ? 0 : man | (uint64_t)implicit << m;
}

/* The value of an encoding (formats.py's FloatFormat.decode): its kind, sign, and for a finite
   value sig * 2**exp, sig not normalized. */
int floatfmt_decode(int *kind, int *neg, bigint *sig, int64_t *exp, const floatfmt *f, const bigint *bits) {
    int64_t m = f->man_bits;
    if (1 + f->exp_bits + m <= 64) {
        uint64_t word;
        floatfmt_decode_word(kind, neg, &word, exp, f, bits->n ? bits->w[0] : 0);
        return big_set_word(sig, word, 0);
    }
    bigint t;
    big_init(&t);
    int rc = big_shr(&t, bits, m, 0), implicit;
    int64_t exp_field = t.n ? (int64_t)(t.w[0] & (((uint64_t)1 << f->exp_bits) - 1)) : 0;
    big_free(&t);
    if (rc || (rc = big_low(sig, bits, m))) return rc;
    *neg = big_bit(bits, f->exp_bits + m);
    classify(kind, neg, exp, &implicit, f, exp_field, !sig->n, big_is_ones(sig, m));
    if (*kind != VALUE_FINITE) {
        sig->n = 0;
        return 0;
    }
    return implicit ? big_set_bit(sig, m) : 0;
}

/* Whether an inexact magnitude's truncated significand rounds away from zero. */
static int rounds_away(int rnd, int neg, int odd, int rest) {
    switch (rnd) {
    case ROUND_NEAREST_EVEN: return rest == REST_HIGH || (rest == REST_HALF && odd);
    case ROUND_NEAREST_AWAY: return rest >= REST_HALF;
    case ROUND_TOWARD_ZERO: return 0;
    case ROUND_TOWARD_POSITIVE: return rest && !neg;
    default: return rest && neg;
    }
}

/* The encoding of a finite value past the largest finite value: an infinity, NaN or the largest
   finite value, by the rounding direction and the overflow mode. */
static int overflow_bits(bigint *bits, const floatfmt *f, int neg, int rnd, int ovf) {
    int outward = rnd == (neg ? ROUND_TOWARD_NEGATIVE : ROUND_TOWARD_POSITIVE);
    int nearest = rnd == ROUND_NEAREST_EVEN || rnd == ROUND_NEAREST_AWAY;
    if (ovf != OVERFLOW_SATURATE) {
        if (f->special == SPECIAL_IEEE) {
            if (nearest || outward) return floatfmt_special(bits, f, ENCODING_INFINITY, neg);
        } else if (outward || (nearest && ovf == OVERFLOW_NAN)) {
            return floatfmt_special(bits, f, ENCODING_NAN, neg);
        }
    }
    return floatfmt_special(bits, f, ENCODING_MAX_FINITE, neg);
}

/* The encoding of a nonzero magnitude below a quarter of the smallest subnormal: the smallest
   subnormal when rounding outward, else zero. */
static int tiny_bits(bigint *bits, const floatfmt *f, int neg, int rnd) {
    if (rnd != (neg ? ROUND_TOWARD_NEGATIVE : ROUND_TOWARD_POSITIVE)) return floatfmt_special(bits, f, ENCODING_ZERO, neg);
    bigint one;
    big_init(&one);
    int rc = big_set_word(&one, 1, 0);
    if (!rc) rc = assemble(bits, f, neg, 0, &one);
    big_free(&one);
    return rc;
}

/* bits = the encoding of (-1)**neg * (sig + rest) * 2**(b - m), where b = max(top, emin) for
   2**top <= the value < 2**(top + 1), so sig < 2**(m + 1); sig is consumed. */
static int finish(bigint *bits, const floatfmt *f, int neg, bigint *sig, int64_t b, int rest, int rnd, int ovf) {
    int64_t m = f->man_bits;
    int rc;
    if (rounds_away(rnd, neg, big_bit(sig, 0), rest)) {
        bigint one;
        big_init(&one);
        if (!(rc = big_set_word(&one, 1, 0))) rc = big_add(sig, sig, &one);
        big_free(&one);
        if (rc) return rc;
        if (big_bits(sig) > m + 1) {
            /* carried into a new bit: sig is 2**(m + 1) */
            if ((rc = big_shr(sig, sig, 1, 0))) return rc;
            b++;
        }
    }
    int64_t exp_field = 0;
    if (big_bits(sig) > m) {
        exp_field = b + f->bias;
        if ((rc = big_low(sig, sig, m))) return rc;
    }
    if (exp_field > f->max_exp_field || (exp_field == f->max_exp_field && f->special == SPECIAL_FN && big_is_ones(sig, m)))
        return overflow_bits(bits, f, neg, rnd, ovf);
    if (!exp_field && !sig->n) return floatfmt_special(bits, f, ENCODING_ZERO, neg);
    return assemble(bits, f, neg, exp_field, sig);
}

/* bits = the encoding of a value (formats.py's FloatFormat.encode and encode_scaled): NaN, an
   infinity, or the finite (-1)**neg * a * 2**e for a >= 0 (zero with the sign neg for a = 0). An
   infinity gives NaN in a format without infinities, or its largest finite value with the
   "saturate" overflow mode. */
int floatfmt_encode(bigint *bits, const floatfmt *f, int kind, int neg, const bigint *a, int64_t e, int rnd, int ovf) {
    if (kind == VALUE_NAN) return floatfmt_special(bits, f, ENCODING_NAN, neg);
    if (kind == VALUE_INFINITE) {
        if (f->special == SPECIAL_IEEE) return floatfmt_special(bits, f, ENCODING_INFINITY, neg);
        return floatfmt_special(bits, f, ovf == OVERFLOW_SATURATE ? ENCODING_MAX_FINITE : ENCODING_NAN, neg);
    }
    if (!a->n) return floatfmt_special(bits, f, ENCODING_ZERO, neg);
    int64_t m = f->man_bits, top = e + big_bits(a) - 1; /* 2**top <= a * 2**e < 2**(top + 1) */
    if (top > f->emax + 1) return overflow_bits(bits, f, neg, rnd, ovf);
    if (top + 1 <= f->emin - m - 2) return tiny_bits(bits, f, neg, rnd);
    int64_t b = top > f->emin ? top : f->emin, q = b - m; /* the binade and its quantum 2**q */
    bigint sig;
    big_init(&sig);
    int rc, rest = REST_ZERO;
    if (e >= q) {
        rc = big_shl(&sig, a, e - q); /* at most m bits: exact */
    } else {
        int64_t s = q - e;
        int half = big_bit(a, s - 1), sticky = big_low_nonzero(a, s - 1);
        rest = half ? (sticky ? REST_HIGH : REST_HALF) : (sticky ? REST_LOW : REST_ZERO);
        rc = big_shr(&sig, a, s, 0);
    }
    sig.neg = 0;
    if (!rc) rc = finish(bits, f, neg, &sig, b, rest, rnd, ovf);
    big_free(&sig);
    return rc;
}

/* The encoding a dyadic rounds to. */
int encode_dyad(bigint *bits, const floatfmt *f, const dyad *x, int rnd, int ovf) {
    bigint a = big_magnitude(&x->m);
    return floatfmt_encode(bits, f, VALUE_FINITE, x->m.neg, &a, x->e, rnd, ovf);
}

/* bits = the encoding of (-1)**neg * a / b for a >= 0 and b > 0. */
int floatfmt_encode_ratio(bigint *bits, const floatfmt *f, int neg, const bigint *a, const bigint *b, int rnd, int ovf) {
    if (!a->n) return floatfmt_special(bits, f, ENCODING_ZERO, neg);
    int64_t m = f->man_bits, t = big_bits(a) - big_bits(b); /* 2**(t - 1) < a / b < 2**(t + 1) */
    if (t - 1 > f->emax + 1) return overflow_bits(bits, f, neg, rnd, ovf);
    if (t + 1 <= f->emin - m - 2) return tiny_bits(bits, f, neg, rnd);
    bigint num, den, sig;
    big_init(&num);
    big_init(&den);
    big_init(&sig);
    /* top with 2**top <= a / b < 2**(top + 1) */
    int rc = t >= 0 ? big_shl(&den, b, t) : big_shl(&num, a, -t);
    int64_t top = t;
    if (!rc && big_cmp_mag(t >= 0 ? a : &num, t >= 0 ? &den : b) < 0) top--;
    int64_t bexp = top > f->emin ? top : f->emin, q = bexp - m;
    /* sig = floor(a / b * 2**(1 - q)): the significand and one more bit */
    if (!rc) rc = 1 - q >= 0 ? big_shl(&num, a, 1 - q) : big_copy(&num, a);
    if (!rc) rc = 1 - q >= 0 ? big_copy(&den, b) : big_shl(&den, b, q - 1);
    int exact = 1;
    if (!rc) rc = big_divmod(&sig, &exact, &num, &den);
    if (!rc) {
        int half = big_bit(&sig, 0);
        int rest = half ? (exact ? REST_HALF : REST_HIGH) : (exact ? REST_ZERO : REST_LOW);
        if (!(rc = big_shr(&sig, &sig, 1, 0))) rc = finish(bits, f, neg, &sig, bexp, rest, rnd, ovf);
    }
    big_free(&num);
    big_free(&den);
    big_free(&sig);
    return rc;
}

/* The values of an integer format of itemsize bytes: two's complement when signed. */
int intfmt_range(bigint *lo, bigint *hi, int64_t itemsize, int is_signed) {
    int64_t bits = 8 * itemsize - (is_signed ? 1 : 0);
    bigint one;
    big_init(&one);
    int rc = big_set_si(&one, -1);
    if (!rc) rc = big_set_word(hi, 1, 0);
    if (!rc) rc = big_shl(hi, hi, bits);
    if (!rc && is_signed) {
        rc = big_copy(lo, hi);
        lo->neg = 1;
    } else if (!rc) {
        lo->n = 0;
        lo->neg = 0;
    }
    if (!rc) rc = big_add(hi, hi, &one);
    big_free(&one);
    return rc;
}

/* v = the integer of an encoding. */
int intfmt_decode(bigint *v, const bigint *bits, int64_t itemsize, int is_signed) {
    int rc = big_copy(v, bits);
    if (rc || !is_signed || !big_bit(bits, 8 * itemsize - 1)) return rc;
    /* v - 2**(8 itemsize) */
    bigint t;
    big_init(&t);
    if (!(rc = big_set_word(&t, 1, 1))) rc = big_shl(&t, &t, 8 * itemsize);
    if (!rc) rc = big_add(v, v, &t);
    big_free(&t);
    return rc;
}

/* bits = the encoding of the integer v (formats.py's IntFormat.encode): out of range, *fits = 0
   with the "error" overflow mode, v clamped to the range with "saturate", and v wrapped modulo
   2**(8 itemsize) with "wrap". */
int intfmt_encode(bigint *bits, int *fits, const bigint *v, int64_t itemsize, int is_signed, int ovf) {
    bigint lo, hi, w;
    big_init(&lo);
    big_init(&hi);
    big_init(&w);
    int rc = intfmt_range(&lo, &hi, itemsize, is_signed);
    const bigint *x = v;
    *fits = 1;
    if (!rc) {
        int below = v->neg && (!lo.n || big_cmp_mag(v, &lo) > 0), above = !v->neg && big_cmp_mag(v, &hi) > 0;
        if ((below || above) && ovf == INT_OVERFLOW_ERROR) *fits = 0;
        if ((below || above) && ovf == INT_OVERFLOW_SATURATE) x = below ? &lo : &hi;
    }
    if (!rc && *fits) {
        int64_t width = 8 * itemsize;
        /* x mod 2**width: |x| mod 2**width, subtracted from 2**width when x < 0 */
        rc = big_low(bits, x, width);
        if (!rc && x->neg && bits->n) {
            bits->neg = 1;
            if (!(rc = big_set_word(&w, 1, 0)) && !(rc = big_shl(&w, &w, width))) rc = big_add(bits, &w, bits);
        }
    }
    big_free(&lo);
    big_free(&hi);
    big_free(&w);
    return rc;
}

/* n = num / den rounded to an integer in the direction rnd, for den > 0. */
int round_integer(bigint *n, const bigint *num, const bigint *den, int rnd) {
    bigint twice, q;
    big_init(&twice);
    big_init(&q);
    int neg = num->neg, exact = 1;
    /* q = floor(2 |num| / den): the magnitude's quotient and one more bit */
    int rc = big_shl(&twice, num, 1);
    if (!rc) rc = big_divmod(&q, &exact, &twice, den);
    if (!rc) {
        int half = big_bit(&q, 0);
        int rest = half ? (exact ? REST_HALF : REST_HIGH) : (exact ? REST_ZERO : REST_LOW);
        rc = big_shr(&q, &q, 1, 0);
        if (!rc && rounds_away(rnd, neg, big_bit(&q, 0), rest)) {
            bigint one;
            big_init(&one);
            if (!(rc = big_set_word(&one, 1, 0))) rc = big_add(&q, &q, &one);
            big_free(&one);
        }
    }
    if (!rc) {
        rc = big_copy(n, &q);
        n->neg = neg && n->n;
    }
    big_free(&twice);
    big_free(&q);
    return rc;
}

/* The largest finite value of f (largest may be NULL) and its overflow threshold to nearest: the
   largest value plus half its ulp, the smallest magnitude that rounds past it. */
int floatfmt_threshold(dyad *theta, dyad *largest, const floatfmt *f) {
    bigint bits;
    big_init(&bits);
    dyad half;
    dyad_init(&half);
    int kind, neg;
    int rc = floatfmt_special(&bits, f, ENCODING_MAX_FINITE, 0);
    if (!rc) rc = floatfmt_decode(&kind, &neg, &theta->m, &theta->e, f, &bits);
    if (!rc && largest) rc = dyad_copy(largest, theta);
    if (!rc) rc = dyad_set_si(&half, 1, f->emax - f->man_bits - 1);
    if (!rc) rc = dyad_add_exact(theta, theta, &half);
    big_free(&bits);
    dyad_free(&half);
    return rc;
}
