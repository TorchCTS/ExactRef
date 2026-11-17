/* The extension module exactref._c: the Python entry points to exactref's C (internal). Only
   exactref._native imports it. Built for the limited API of Python 3.10 (abi3): arrays cross as
   bytes (no buffer protocol there), integers of up to two words through the int API and longer
   ones through int.to_bytes and int.from_bytes. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "exactref.h"
#include "words.h"

/* Entry points return None as Py_NewRef(Py_None), never by Py_RETURN_NONE: Python 3.12.3's headers define that macro
   without taking a reference even for the 3.10 limited API, so a module built with them would drop a reference to None
   on every call under 3.10 and 3.11, which ends the interpreter. */

/* Dyadic exponents up to this size cross as 64-bit integers; beyond it a call returns
   NotImplemented and _native moves the exponents to a base first. */
#define EXP_CROSS ((long long)1 << 61)

static PyObject *little_obj;     /* the str "little" */
static PyObject *word_bits_obj;  /* the int 64 */
static PyObject *from_bytes_obj; /* int.from_bytes */

static PyObject *py_physical_cores(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyLong_FromLong(physical_cores());
}

/* Sets the Python error for rc and returns NULL. */
static PyObject *error(int rc) {
    switch (rc) {
    case EXACTREF_NOMEM: return PyErr_NoMemory();
    case EXACTREF_OVERFLOW: PyErr_SetString(PyExc_OverflowError, "exactref._c: a value is past the range the C holds"); return NULL;
    case EXACTREF_ZERO_DIVISION: PyErr_SetString(PyExc_ZeroDivisionError, "dyadic division by zero"); return NULL;
    case EXACTREF_DOMAIN: PyErr_SetString(PyExc_ValueError, "square root of a negative number"); return NULL;
    case EXACTREF_LOG_DOMAIN: PyErr_SetString(PyExc_ValueError, "logarithm of a number that is not positive"); return NULL;
    case EXACTREF_INTERNAL: PyErr_SetString(PyExc_SystemError, "exactref._c: an internal check failed"); return NULL;
    case EXACTREF_INT_OVERFLOW: PyErr_SetString(PyExc_OverflowError, "an integer result is outside the output format's range"); return NULL;
    default: PyErr_SetString(PyExc_SystemError, "exactref._c: unknown error"); return NULL;
    }
}

/* b = the Python int obj; -1 with a Python error set on failure. */
static int int_load(PyObject *obj, bigint *b) {
    int overflow;
    long long v = PyLong_AsLongLongAndOverflow(obj, &overflow);
    if (v == -1 && PyErr_Occurred()) return -1;
    b->n = 0;
    if (!overflow) {
        uint64_t mag = v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v;
        b->neg = v < 0;
        if (mag) b->w[b->n++] = mag;
        return 0;
    }
    b->neg = overflow < 0;
    int rc = -1;
    PyObject *mag = PyNumber_Absolute(obj), *bits = NULL, *bytes = NULL;
    if (!mag) return -1;
    /* two words: the low one by its mask, the high one after a shift */
    PyObject *high = PyNumber_Rshift(mag, word_bits_obj);
    if (!high) {
        Py_DECREF(mag);
        return -1;
    }
    int big = 0;
    unsigned long long hi = PyLong_AsUnsignedLongLong(high);
    Py_DECREF(high);
    if (hi == (unsigned long long)-1 && PyErr_Occurred()) {
        if (!PyErr_ExceptionMatches(PyExc_OverflowError)) {
            Py_DECREF(mag);
            return -1;
        }
        PyErr_Clear();
        big = 1;
    }
    if (!big) {
        b->w[0] = PyLong_AsUnsignedLongLongMask(mag);
        b->w[1] = hi;
        b->n = 2;
        big_trim(b);
        Py_DECREF(mag);
        return 0;
    }
    bits = PyObject_CallMethod(mag, "bit_length", NULL);
    Py_ssize_t nbits = bits ? PyLong_AsSsize_t(bits) : -1;
    if (nbits >= 0) bytes = PyObject_CallMethod(mag, "to_bytes", "nO", (nbits + 7) / 8, little_obj);
    char *s;
    Py_ssize_t len;
    if (bytes && PyBytes_AsStringAndSize(bytes, &s, &len) == 0) {
        size_t words = ((size_t)len + 7) / 8;
        if (big_reserve(b, words)) {
            PyErr_NoMemory();
        } else {
            for (size_t i = 0; i < words; i++) {
                uint64_t w = 0;
                for (size_t k = 0; k < 8 && 8 * i + k < (size_t)len; k++) w |= (uint64_t)(unsigned char)s[8 * i + k] << (8 * k);
                b->w[i] = w;
            }
            b->n = words;
            big_trim(b);
            rc = 0;
        }
    }
    Py_XDECREF(mag);
    Py_XDECREF(bits);
    Py_XDECREF(bytes);
    return rc;
}

/* The Python int of b. */
static PyObject *int_store(const bigint *b) {
    if (!b->n) return PyLong_FromLong(0);
    if (b->n == 1 && b->w[0] <= (uint64_t)LLONG_MAX) return PyLong_FromLongLong(b->neg ? -(long long)b->w[0] : (long long)b->w[0]);
    PyObject *mag;
    if (b->n <= 2) {
        /* the high word shifted up, or'd with the low one */
        PyObject *high = PyLong_FromUnsignedLongLong(b->n == 2 ? b->w[1] : 0), *low = PyLong_FromUnsignedLongLong(b->w[0]);
        PyObject *shifted = high ? PyNumber_Lshift(high, word_bits_obj) : NULL;
        mag = shifted && low ? PyNumber_Or(shifted, low) : NULL;
        Py_XDECREF(high);
        Py_XDECREF(low);
        Py_XDECREF(shifted);
    } else {
        PyObject *bytes = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(8 * b->n));
        if (!bytes) return NULL;
        char *s = PyBytes_AsString(bytes); /* a new object: still ours to fill */
        for (size_t i = 0; i < 8 * b->n; i++) s[i] = (char)(b->w[i / 8] >> (8 * (i % 8)));
        mag = PyObject_CallFunctionObjArgs(from_bytes_obj, bytes, little_obj, NULL);
        Py_DECREF(bytes);
    }
    if (!mag || !b->neg) return mag;
    PyObject *negated = PyNumber_Negative(mag);
    Py_DECREF(mag);
    return negated;
}

/* x = the pair (m, e); 1 when e is beyond EXP_CROSS, -1 with a Python error on failure. */
static int dyad_load(PyObject *pair, dyad *x) {
    if (!PyTuple_Check(pair) || PyTuple_Size(pair) != 2) {
        PyErr_SetString(PyExc_TypeError, "a dyadic is a pair (m, e)");
        return -1;
    }
    int overflow;
    long long e = PyLong_AsLongLongAndOverflow(PyTuple_GetItem(pair, 1), &overflow);
    if (e == -1 && PyErr_Occurred()) return -1;
    if (overflow || e > EXP_CROSS || e < -EXP_CROSS) return 1;
    int rc = int_load(PyTuple_GetItem(pair, 0), &x->m);
    x->e = x->m.n ? e : 0; /* a zero is 0 2**0 */
    return rc;
}

/* The dyadic operations' error for rc: NotImplemented (not an error) for an exponent out of range, which _native makes
   again with the exponents moved to a base; else error(rc). */
static PyObject *crossing_error(int rc) { return rc == EXACTREF_OVERFLOW ? Py_NewRef(Py_NotImplemented) : error(rc); }

/* The pair (m, e) of r, or the dyadic operations' error of rc. */
static PyObject *dyad_store(const dyad *r, int rc) {
    if (rc) return crossing_error(rc);
    PyObject *m = int_store(&r->m);
    if (!m) return NULL;
    PyObject *e = PyLong_FromLongLong(r->e);
    PyObject *out = e ? PyTuple_Pack(2, m, e) : NULL;
    Py_DECREF(m);
    Py_XDECREF(e);
    return out;
}

typedef int unary_op(dyad *, const dyad *, int64_t, int);
typedef int binary_op(dyad *, const dyad *, const dyad *, int64_t, int);

static PyObject *unary(PyObject *args, unary_op *op) {
    PyObject *xo;
    long long p;
    int up;
    if (!PyArg_ParseTuple(args, "OLp", &xo, &p, &up)) return NULL;
    dyad x, r;
    dyad_init(&x);
    dyad_init(&r);
    PyObject *out;
    int loaded = dyad_load(xo, &x);
    if (loaded) {
        out = loaded > 0 ? crossing_error(EXACTREF_OVERFLOW) : NULL;
    } else {
        out = dyad_store(&r, op(&r, &x, p, up));
    }
    dyad_free(&x);
    dyad_free(&r);
    return out;
}

static PyObject *binary(PyObject *args, binary_op *op) {
    PyObject *xo, *yo;
    long long p;
    int up;
    if (!PyArg_ParseTuple(args, "OOLp", &xo, &yo, &p, &up)) return NULL;
    dyad x, y, r;
    dyad_init(&x);
    dyad_init(&y);
    dyad_init(&r);
    PyObject *out;
    int loaded = dyad_load(xo, &x);
    if (!loaded) loaded = dyad_load(yo, &y);
    if (loaded) {
        out = loaded > 0 ? crossing_error(EXACTREF_OVERFLOW) : NULL;
    } else {
        out = dyad_store(&r, op(&r, &x, &y, p, up));
    }
    dyad_free(&x);
    dyad_free(&y);
    dyad_free(&r);
    return out;
}

static PyObject *py_rnd(PyObject *self, PyObject *args) {
    (void)self;
    return unary(args, dyad_rnd);
}

/* expm1_end(x, p, up), for tests: OverflowError for an x with an exponent beyond 2**61, as exp's binding */
static PyObject *py_expm1_end(PyObject *self, PyObject *args) {
    PyObject *xo;
    long long p;
    int up;
    (void)self;
    if (!PyArg_ParseTuple(args, "OLp", &xo, &p, &up)) return NULL;
    dyad x, r;
    dyad_init(&x);
    dyad_init(&r);
    PyObject *out = NULL;
    int loaded = dyad_load(xo, &x);
    if (loaded > 0)
        PyErr_SetString(PyExc_OverflowError, "expm1 of a dyadic with an exponent beyond 2**61");
    else if (!loaded)
        out = dyad_store(&r, expm1_end(&r, &x, p, up));
    dyad_free(&x);
    dyad_free(&r);
    return out;
}

static PyObject *py_sqrt(PyObject *self, PyObject *args) {
    (void)self;
    return unary(args, dyad_sqrt);
}

static PyObject *py_add(PyObject *self, PyObject *args) {
    (void)self;
    return binary(args, dyad_add);
}

static PyObject *py_mul(PyObject *self, PyObject *args) {
    (void)self;
    return binary(args, dyad_mul);
}

static PyObject *py_div(PyObject *self, PyObject *args) {
    (void)self;
    return binary(args, dyad_div);
}

/* (lo, hi): two pairs, or NULL with the error of rc. */
static PyObject *pair_of_dyads(const dyad *lo, const dyad *hi, int rc) {
    if (rc) return error(rc);
    PyObject *a = dyad_store(lo, 0), *b = a ? dyad_store(hi, 0) : NULL;
    PyObject *out = b ? PyTuple_Pack(2, a, b) : NULL;
    Py_XDECREF(a);
    Py_XDECREF(b);
    return out;
}

static PyObject *py_ln2(PyObject *self, PyObject *args) {
    long long p;
    (void)self;
    if (!PyArg_ParseTuple(args, "L", &p)) return NULL;
    dyad lo, hi;
    dyad_init(&lo);
    dyad_init(&hi);
    PyObject *out = pair_of_dyads(&lo, &hi, ln2_bounds(&lo, &hi, p));
    dyad_free(&lo);
    dyad_free(&hi);
    return out;
}

static PyObject *py_exp(PyObject *self, PyObject *args) {
    PyObject *xo;
    long long p;
    (void)self;
    if (!PyArg_ParseTuple(args, "OL", &xo, &p)) return NULL;
    dyad x, lo, hi;
    bigint n;
    dyad_init(&x);
    dyad_init(&lo);
    dyad_init(&hi);
    big_init(&n);
    PyObject *out = NULL;
    int loaded = dyad_load(xo, &x);
    if (loaded > 0) {
        PyErr_SetString(PyExc_OverflowError, "exp of a dyadic with an exponent beyond 2**61");
    } else if (!loaded) {
        PyObject *bounds = pair_of_dyads(&lo, &hi, exp_bounds(&lo, &hi, &n, &x, p));
        PyObject *shift = bounds ? int_store(&n) : NULL;
        if (shift) out = PyTuple_Pack(3, PyTuple_GetItem(bounds, 0), PyTuple_GetItem(bounds, 1), shift);
        Py_XDECREF(bounds);
        Py_XDECREF(shift);
    }
    dyad_free(&x);
    dyad_free(&lo);
    dyad_free(&hi);
    big_free(&n);
    return out;
}

static PyObject *py_exp_near_one(PyObject *self, PyObject *args) {
    PyObject *xo;
    long long p;
    (void)self;
    if (!PyArg_ParseTuple(args, "OL", &xo, &p)) return NULL;
    dyad x, lo, hi;
    dyad_init(&x);
    dyad_init(&lo);
    dyad_init(&hi);
    PyObject *out = NULL;
    int loaded = dyad_load(xo, &x), ok_lo = 0, ok_hi = 0, rc = 0;
    if (loaded > 0) {
        PyErr_SetString(PyExc_OverflowError, "exp of a dyadic with an exponent beyond 2**61");
    } else if (!loaded) {
        if (!(rc = exp_near_one(&lo, &ok_lo, &x, p, 0))) rc = exp_near_one(&hi, &ok_hi, &x, p, 1);
        if (!rc && !(ok_lo && ok_hi)) {
            Py_INCREF(Py_None);
            out = Py_None;
        } else {
            out = pair_of_dyads(&lo, &hi, rc);
        }
    }
    dyad_free(&x);
    dyad_free(&lo);
    dyad_free(&hi);
    return out;
}

static PyObject *py_log(PyObject *self, PyObject *args) {
    PyObject *mo, *eo;
    long long p;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOL", &mo, &eo, &p)) return NULL;
    bigint m, e;
    dyad lo, hi;
    big_init(&m);
    big_init(&e);
    dyad_init(&lo);
    dyad_init(&hi);
    PyObject *out = NULL;
    if (!int_load(mo, &m) && !int_load(eo, &e)) out = pair_of_dyads(&lo, &hi, log_bounds(&lo, &hi, &m, &e, p));
    big_free(&m);
    big_free(&e);
    dyad_free(&lo);
    dyad_free(&hi);
    return out;
}

static PyObject *py_pi(PyObject *self, PyObject *args) {
    long long p;
    (void)self;
    if (!PyArg_ParseTuple(args, "L", &p)) return NULL;
    dyad lo, hi;
    dyad_init(&lo);
    dyad_init(&hi);
    PyObject *out = pair_of_dyads(&lo, &hi, pi_bounds(&lo, &hi, p));
    dyad_free(&lo);
    dyad_free(&hi);
    return out;
}

static PyObject *py_cos_sin(PyObject *self, PyObject *args) {
    PyObject *xo;
    long long p;
    (void)self;
    if (!PyArg_ParseTuple(args, "OL", &xo, &p)) return NULL;
    dyad x, c[2], s[2];
    dyad_init(&x);
    for (int k = 0; k < 2; k++) {
        dyad_init(&c[k]);
        dyad_init(&s[k]);
    }
    PyObject *out = NULL;
    int loaded = dyad_load(xo, &x);
    if (loaded > 0) {
        PyErr_SetString(PyExc_OverflowError, "cos and sin of a dyadic with an exponent beyond 2**61");
    } else if (!loaded) {
        int rc = cos_sin_bounds(c, s, &x, p);
        PyObject *a = pair_of_dyads(&c[0], &c[1], rc), *b = a ? pair_of_dyads(&s[0], &s[1], 0) : NULL;
        out = b ? PyTuple_Pack(2, a, b) : NULL;
        Py_XDECREF(a);
        Py_XDECREF(b);
    }
    dyad_free(&x);
    for (int k = 0; k < 2; k++) {
        dyad_free(&c[k]);
        dyad_free(&s[k]);
    }
    return out;
}

static PyObject *py_cos_sin_range(PyObject *self, PyObject *args) {
    PyObject *loo, *hio;
    long long p;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOL", &loo, &hio, &p)) return NULL;
    dyad lo, hi, c[2], s[2];
    dyad_init(&lo);
    dyad_init(&hi);
    for (int k = 0; k < 2; k++) {
        dyad_init(&c[k]);
        dyad_init(&s[k]);
    }
    PyObject *out = NULL;
    int a = dyad_load(loo, &lo), b = a ? 0 : dyad_load(hio, &hi);
    if (a > 0 || b > 0) {
        PyErr_SetString(PyExc_OverflowError, "cos and sin of a dyadic with an exponent beyond 2**61");
    } else if (!a && !b) {
        int rc = dyad_cmp(&lo, &hi) > 0 ? EXACTREF_DOMAIN : cos_sin_range(c, s, &lo, &hi, p);
        PyObject *x = pair_of_dyads(&c[0], &c[1], rc), *y = x ? pair_of_dyads(&s[0], &s[1], 0) : NULL;
        out = y ? PyTuple_Pack(2, x, y) : NULL;
        Py_XDECREF(x);
        Py_XDECREF(y);
    }
    dyad_free(&lo);
    dyad_free(&hi);
    for (int k = 0; k < 2; k++) {
        dyad_free(&c[k]);
        dyad_free(&s[k]);
    }
    return out;
}

/* An exponent as a 64-bit integer, clamped to [-EXP_CROSS, EXP_CROSS]: a format's encodings and
   roundings need no more, since every format lies within 2**+-EXACT_LIMIT and a significand has
   far fewer than 2**60 bits. -1 with a Python error on failure. */
static int exp_load(PyObject *obj, int64_t *e) {
    int overflow;
    long long v = PyLong_AsLongLongAndOverflow(obj, &overflow);
    if (v == -1 && PyErr_Occurred()) return -1;
    if (overflow) v = overflow > 0 ? EXP_CROSS : -EXP_CROSS;
    *e = v > EXP_CROSS ? EXP_CROSS : v < -EXP_CROSS ? -EXP_CROSS : v;
    return 0;
}

/* f = the format of formats.py's layout tuple (exp_bits, man_bits, special, bias, nan_pos,
   nan_neg): bias None for the default, the NaN encodings None unless the definition gives them
   (held in nan). -1 with a Python error on failure. */
static int layout_load(PyObject *layout, floatfmt *f, bigint nan[2]) {
    long long e, m;
    int special;
    PyObject *bias_obj, *pos, *neg;
    if (!PyArg_ParseTuple(layout, "LLiOOO", &e, &m, &special, &bias_obj, &pos, &neg)) return -1;
    int64_t bias = 0;
    if (bias_obj != Py_None) {
        bias = PyLong_AsLongLong(bias_obj);
        if (bias == -1 && PyErr_Occurred()) return -1;
    }
    if (floatfmt_set(f, e, m, special, bias_obj != Py_None ? &bias : NULL, 0)) {
        PyErr_SetString(PyExc_SystemError, "exactref._c: an invalid format layout");
        return -1;
    }
    if (pos != Py_None) {
        if (int_load(pos, &nan[0]) || int_load(neg, &nan[1])) return -1;
        f->nan[0] = &nan[0];
        f->nan[1] = &nan[1];
    }
    return 0;
}

static PyObject *py_float_layout(PyObject *self, PyObject *args) {
    long long e, m;
    int special;
    PyObject *bias_obj;
    (void)self;
    if (!PyArg_ParseTuple(args, "LLiO", &e, &m, &special, &bias_obj)) return NULL;
    int64_t bias = 0;
    if (bias_obj != Py_None) {
        int overflow;
        bias = PyLong_AsLongLongAndOverflow(bias_obj, &overflow);
        if (bias == -1 && PyErr_Occurred()) return NULL;
        if (overflow) return PyUnicode_FromString("range");
    }
    floatfmt f;
    int rc = floatfmt_set(&f, e, m, special, bias_obj != Py_None ? &bias : NULL, 1);
    if (rc)
        return PyUnicode_FromString(rc == EXACTREF_FORMAT_BITS ? "bits" : rc == EXACTREF_FORMAT_BYTES ? "bytes" : rc == EXACTREF_FORMAT_SIZE ? "size" : "range");
    bigint sign;
    big_init(&sign);
    if (!(rc = big_set_word(&sign, 1, 0))) rc = big_shl(&sign, &sign, e + m);
    PyObject *sign_obj = rc ? error(rc) : int_store(&sign);
    big_free(&sign);
    if (!sign_obj) return NULL;
    return Py_BuildValue("(LLLLN)", (long long)f.bias, (1 + e + m) / 8, (long long)f.emin, (long long)f.emax, sign_obj);
}

static PyObject *py_float_decode(PyObject *self, PyObject *args) {
    PyObject *layout, *bits_obj;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &layout, &bits_obj)) return NULL;
    floatfmt f;
    bigint nan[2], bits, sig;
    big_init(&nan[0]);
    big_init(&nan[1]);
    big_init(&bits);
    big_init(&sig);
    PyObject *out = NULL;
    if (!layout_load(layout, &f, nan) && !int_load(bits_obj, &bits)) {
        int kind, neg;
        int64_t exp;
        int rc = floatfmt_decode(&kind, &neg, &sig, &exp, &f, &bits);
        PyObject *s = rc ? error(rc) : int_store(&sig);
        if (s) out = Py_BuildValue("(iNNL)", kind, PyBool_FromLong(neg), s, (long long)exp);
    }
    big_free(&nan[0]);
    big_free(&nan[1]);
    big_free(&bits);
    big_free(&sig);
    return out;
}

static PyObject *py_float_encode(PyObject *self, PyObject *args) {
    PyObject *layout, *m_obj, *e_obj;
    int kind, neg, rnd, ovf;
    (void)self;
    if (!PyArg_ParseTuple(args, "OipOOii", &layout, &kind, &neg, &m_obj, &e_obj, &rnd, &ovf)) return NULL;
    floatfmt f;
    bigint nan[2], m, bits;
    big_init(&nan[0]);
    big_init(&nan[1]);
    big_init(&m);
    big_init(&bits);
    int64_t e;
    PyObject *out = NULL;
    if (!layout_load(layout, &f, nan) && !int_load(m_obj, &m) && !exp_load(e_obj, &e)) {
        m.neg = 0;
        int rc = floatfmt_encode(&bits, &f, kind, neg, &m, e, rnd, ovf);
        out = rc ? error(rc) : int_store(&bits);
    }
    big_free(&nan[0]);
    big_free(&nan[1]);
    big_free(&m);
    big_free(&bits);
    return out;
}

static PyObject *py_float_encode_ratio(PyObject *self, PyObject *args) {
    PyObject *layout, *a_obj, *b_obj;
    int neg, rnd, ovf;
    (void)self;
    if (!PyArg_ParseTuple(args, "OpOOii", &layout, &neg, &a_obj, &b_obj, &rnd, &ovf)) return NULL;
    floatfmt f;
    bigint nan[2], a, b, bits;
    big_init(&nan[0]);
    big_init(&nan[1]);
    big_init(&a);
    big_init(&b);
    big_init(&bits);
    PyObject *out = NULL;
    if (!layout_load(layout, &f, nan) && !int_load(a_obj, &a) && !int_load(b_obj, &b)) {
        int rc = floatfmt_encode_ratio(&bits, &f, neg, &a, &b, rnd, ovf);
        out = rc ? error(rc) : int_store(&bits);
    }
    big_free(&nan[0]);
    big_free(&nan[1]);
    big_free(&a);
    big_free(&b);
    big_free(&bits);
    return out;
}

static PyObject *py_float_special(PyObject *self, PyObject *args) {
    PyObject *layout;
    int which, neg;
    (void)self;
    if (!PyArg_ParseTuple(args, "Oip", &layout, &which, &neg)) return NULL;
    floatfmt f;
    bigint nan[2], bits;
    big_init(&nan[0]);
    big_init(&nan[1]);
    big_init(&bits);
    PyObject *out = NULL;
    if (!layout_load(layout, &f, nan)) {
        int rc = floatfmt_special(&bits, &f, which, neg);
        out = rc ? error(rc) : int_store(&bits);
    }
    big_free(&nan[0]);
    big_free(&nan[1]);
    big_free(&bits);
    return out;
}

static PyObject *py_float_narrow(PyObject *self, PyObject *args) {
    PyObject *data, *wide_obj, *narrow_obj;
    int big_endian, rnd, ovf;
    (void)self;
    if (!PyArg_ParseTuple(args, "SpOOii", &data, &big_endian, &wide_obj, &narrow_obj, &rnd, &ovf)) return NULL;
    floatfmt wide, narrow;
    bigint wide_nan[2], narrow_nan[2], bits, sig;
    big_init(&wide_nan[0]);
    big_init(&wide_nan[1]);
    big_init(&narrow_nan[0]);
    big_init(&narrow_nan[1]);
    big_init(&bits);
    big_init(&sig);
    PyObject *out = NULL;
    char *src;
    Py_ssize_t len;
    if (!layout_load(wide_obj, &wide, wide_nan) && !layout_load(narrow_obj, &narrow, narrow_nan) &&
        !PyBytes_AsStringAndSize(data, &src, &len) && (out = PyBytes_FromStringAndSize(NULL, len))) {
        unsigned char *dst = (unsigned char *)PyBytes_AsString(out); /* a new object: still ours to fill */
        size_t itemsize = (size_t)(1 + wide.exp_bits + wide.man_bits) / 8;
        int rc = 0;
        for (size_t i = 0; !rc && i < (size_t)len / itemsize; i++) {
            int kind, neg;
            int64_t exp;
            if ((rc = bytes_load(&bits, (const unsigned char *)src + i * itemsize, itemsize, big_endian)) ||
                (rc = floatfmt_decode(&kind, &neg, &sig, &exp, &wide, &bits)))
                break;
            /* a finite value is rounded to the narrow format and written back exactly; NaN
               (with its payload) and infinities stay */
            if (kind == VALUE_FINITE &&
                ((rc = floatfmt_encode(&bits, &narrow, kind, neg, &sig, exp, rnd, ovf)) ||
                 (rc = floatfmt_decode(&kind, &neg, &sig, &exp, &narrow, &bits)) ||
                 (rc = floatfmt_encode(&bits, &wide, kind, neg, &sig, exp, ROUND_NEAREST_EVEN, OVERFLOW_IEEE))))
                break;
            bytes_store(dst + i * itemsize, itemsize, &bits);
        }
        if (rc) {
            Py_DECREF(out);
            out = error(rc);
        }
    }
    big_free(&wide_nan[0]);
    big_free(&wide_nan[1]);
    big_free(&narrow_nan[0]);
    big_free(&narrow_nan[1]);
    big_free(&bits);
    big_free(&sig);
    return out;
}

static PyObject *py_int_range(PyObject *self, PyObject *args) {
    long long itemsize;
    int is_signed;
    (void)self;
    if (!PyArg_ParseTuple(args, "Lp", &itemsize, &is_signed)) return NULL;
    if (itemsize > 8) return PyUnicode_FromString("size"); /* parts of at most 64 bits */
    bigint lo, hi;
    big_init(&lo);
    big_init(&hi);
    int rc = intfmt_range(&lo, &hi, itemsize, is_signed);
    PyObject *a = rc ? error(rc) : int_store(&lo), *b = a ? int_store(&hi) : NULL;
    PyObject *out = b ? PyTuple_Pack(2, a, b) : NULL;
    Py_XDECREF(a);
    Py_XDECREF(b);
    big_free(&lo);
    big_free(&hi);
    return out;
}

static PyObject *py_int_decode(PyObject *self, PyObject *args) {
    PyObject *bits_obj;
    long long itemsize;
    int is_signed;
    (void)self;
    if (!PyArg_ParseTuple(args, "OLp", &bits_obj, &itemsize, &is_signed)) return NULL;
    bigint bits, v;
    big_init(&bits);
    big_init(&v);
    PyObject *out = NULL;
    if (!int_load(bits_obj, &bits)) {
        int rc = intfmt_decode(&v, &bits, itemsize, is_signed);
        out = rc ? error(rc) : int_store(&v);
    }
    big_free(&bits);
    big_free(&v);
    return out;
}

static PyObject *py_int_encode(PyObject *self, PyObject *args) {
    PyObject *v_obj;
    long long itemsize;
    int is_signed, ovf;
    (void)self;
    if (!PyArg_ParseTuple(args, "OLpi", &v_obj, &itemsize, &is_signed, &ovf)) return NULL;
    bigint v, bits;
    big_init(&v);
    big_init(&bits);
    PyObject *out = NULL;
    if (!int_load(v_obj, &v)) {
        int fits, rc = intfmt_encode(&bits, &fits, &v, itemsize, is_signed, ovf);
        if (rc) {
            out = error(rc);
        } else if (!fits) {
            Py_INCREF(Py_None);
            out = Py_None;
        } else {
            out = int_store(&bits);
        }
    }
    big_free(&v);
    big_free(&bits);
    return out;
}

static PyObject *py_round_integer(PyObject *self, PyObject *args) {
    PyObject *num_obj, *den_obj;
    int rnd;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOi", &num_obj, &den_obj, &rnd)) return NULL;
    bigint num, den, n;
    big_init(&num);
    big_init(&den);
    big_init(&n);
    PyObject *out = NULL;
    if (!int_load(num_obj, &num) && !int_load(den_obj, &den)) {
        int rc = round_integer(&n, &num, &den, rnd);
        out = rc ? error(rc) : int_store(&n);
    }
    big_free(&num);
    big_free(&den);
    big_free(&n);
    return out;
}

/* x = the pair (m, e), its exponent clamped as exp_load's; -1 with a Python error on failure. */
static int dyad_load_clamped(PyObject *pair, dyad *x) {
    if (!PyTuple_Check(pair) || PyTuple_Size(pair) != 2) {
        PyErr_SetString(PyExc_TypeError, "a dyadic is a pair (m, e)");
        return -1;
    }
    if (exp_load(PyTuple_GetItem(pair, 1), &x->e)) return -1;
    int rc = int_load(PyTuple_GetItem(pair, 0), &x->m);
    if (!x->m.n) x->e = 0; /* a zero is 0 2**0 */
    return rc;
}

/* The entry as the pair (flag, midpoint bytes). */
static PyObject *entry_store(const entry *x) {
    PyObject *mid = PyBytes_FromStringAndSize((const char *)x->mid, ENTRY_WIDTH);
    if (!mid) return NULL;
    PyObject *out = Py_BuildValue("(iN)", x->flag, mid);
    return out;
}

/* x = the entry of (flag, midpoint bytes); -1 with a Python error on failure. */
static int entry_load(PyObject *flag_obj, PyObject *mid_obj, entry *x) {
    long flag = PyLong_AsLong(flag_obj);
    if (flag == -1 && PyErr_Occurred()) return -1;
    char *s;
    Py_ssize_t len;
    if (PyBytes_AsStringAndSize(mid_obj, &s, &len)) return -1;
    if (len != ENTRY_WIDTH) {
        PyErr_SetString(PyExc_ValueError, "an entry's midpoint is 24 bytes");
        return -1;
    }
    x->flag = (unsigned char)flag;
    memcpy(x->mid, s, ENTRY_WIDTH);
    return 0;
}

static PyObject *py_exact_entry(PyObject *self, PyObject *args) {
    PyObject *m_obj, *e_obj;
    int kind, neg;
    (void)self;
    if (!PyArg_ParseTuple(args, "ipOO", &kind, &neg, &m_obj, &e_obj)) return NULL;
    bigint m;
    big_init(&m);
    int64_t e;
    entry x;
    PyObject *out = NULL;
    if (!int_load(m_obj, &m) && !exp_load(e_obj, &e)) {
        m.neg = 0;
        int rc = entry_value(&x, kind, neg, &m, e);
        out = rc ? error(rc) : entry_store(&x);
    }
    big_free(&m);
    return out;
}

static PyObject *py_ratio_entry(PyObject *self, PyObject *args) {
    PyObject *num_obj, *den_obj;
    int neg;
    (void)self;
    if (!PyArg_ParseTuple(args, "pOO", &neg, &num_obj, &den_obj)) return NULL;
    bigint num, den;
    big_init(&num);
    big_init(&den);
    entry x;
    PyObject *out = NULL;
    if (!int_load(num_obj, &num) && !int_load(den_obj, &den)) {
        int rc = entry_ratio(&x, neg, &num, &den);
        out = rc ? error(rc) : entry_store(&x);
    }
    big_free(&num);
    big_free(&den);
    return out;
}

static PyObject *py_interval_entry(PyObject *self, PyObject *args) {
    PyObject *lo_obj, *hi_obj;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &lo_obj, &hi_obj)) return NULL;
    dyad lo, hi;
    dyad_init(&lo);
    dyad_init(&hi);
    entry x;
    PyObject *out = NULL;
    if (!dyad_load_clamped(lo_obj, &lo) && !dyad_load_clamped(hi_obj, &hi)) {
        int wide, rc = entry_interval(&x, &wide, &lo, &hi);
        if (rc) {
            out = error(rc);
        } else if (wide) {
            Py_INCREF(Py_None);
            out = Py_None;
        } else {
            out = entry_store(&x);
        }
    }
    dyad_free(&lo);
    dyad_free(&hi);
    return out;
}

static PyObject *py_entry_read(PyObject *self, PyObject *args) {
    PyObject *flag_obj, *mid_obj;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &flag_obj, &mid_obj)) return NULL;
    entry x;
    bigint m, rm;
    big_init(&m);
    big_init(&rm);
    PyObject *out = NULL;
    if (!entry_load(flag_obj, mid_obj, &x)) {
        int kind, neg, out_of_range;
        int64_t e, re;
        int rc = entry_read(&kind, &neg, &m, &e, &rm, &re, &out_of_range, &x);
        PyObject *a = rc ? error(rc) : int_store(&m), *b = a ? int_store(&rm) : NULL;
        if (b) out = Py_BuildValue("(iNNLNLN)", kind, PyBool_FromLong(neg), a, (long long)e, b, (long long)re, PyBool_FromLong(out_of_range));
        else Py_XDECREF(a);
    }
    big_free(&m);
    big_free(&rm);
    return out;
}

static PyObject *py_entry_bounds(PyObject *self, PyObject *args) {
    PyObject *flag_obj, *mid_obj;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &flag_obj, &mid_obj)) return NULL;
    entry x;
    dyad lo, hi;
    dyad_init(&lo);
    dyad_init(&hi);
    PyObject *out = NULL;
    if (!entry_load(flag_obj, mid_obj, &x)) {
        int none, rc = entry_bounds(&lo, &hi, &none, &x);
        if (!rc && none) {
            Py_INCREF(Py_None);
            out = Py_None;
        } else {
            out = pair_of_dyads(&lo, &hi, rc);
        }
    }
    dyad_free(&lo);
    dyad_free(&hi);
    return out;
}

/* p = the part format of a layout: (itemsize, signed) for an integer format, else a float
   format's layout tuple (layout_load). -1 with a Python error on failure. */
static int partfmt_load(PyObject *layout, partfmt *p, bigint nan[2]) {
    if (PyTuple_Check(layout) && PyTuple_Size(layout) == 2) {
        long long itemsize;
        p->is_int = 1;
        if (!PyArg_ParseTuple(layout, "Lp", &itemsize, &p->is_signed)) return -1;
        p->itemsize = itemsize;
        return 0;
    }
    p->is_int = 0;
    return layout_load(layout, &p->f, nan);
}

static void value_init(value *x) { dyad_init(&x->v); }

static void value_free(value *x) { dyad_free(&x->v); }

/* (tolerance, low, high, flags), or the error of rc. */
static PyObject *bounds_store(const bigint *tol, const bigint *low, const bigint *high, int flags, int rc) {
    if (rc) return error(rc);
    PyObject *a = int_store(tol), *b = a ? int_store(low) : NULL, *c = b ? int_store(high) : NULL;
    PyObject *out = c ? Py_BuildValue("(OOOi)", a, b, c, flags) : NULL;
    Py_XDECREF(a);
    Py_XDECREF(b);
    Py_XDECREF(c);
    return out;
}

static PyObject *py_final_bounds(PyObject *self, PyObject *args) {
    PyObject *layout, *f64_layout, *lo_obj, *hi_obj, *ref_obj;
    int rnd, ovf, possible;
    long long w;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOiiOOOLi", &layout, &f64_layout, &rnd, &ovf, &lo_obj, &hi_obj, &ref_obj, &w, &possible)) return NULL;
    floatfmt f, f64;
    bigint nan[2], f64_nan[2], ref, tol, low, high;
    dyad lo, hi;
    bigint *bigs[] = {&nan[0], &nan[1], &f64_nan[0], &f64_nan[1], &ref, &tol, &low, &high};
    for (size_t i = 0; i < sizeof bigs / sizeof *bigs; i++) big_init(bigs[i]);
    dyad_init(&lo);
    dyad_init(&hi);
    PyObject *out = NULL;
    int bounded = lo_obj != Py_None;
    if (!layout_load(layout, &f, nan) && !layout_load(f64_layout, &f64, f64_nan) && !int_load(ref_obj, &ref) &&
        (!bounded || (!dyad_load_clamped(lo_obj, &lo) && !dyad_load_clamped(hi_obj, &hi)))) {
        int flags, rc = final_bounds(&tol, &low, &high, &flags, &f, &f64, rnd, ovf, bounded ? &lo : NULL, bounded ? &hi : NULL, &ref, w, possible);
        out = bounds_store(&tol, &low, &high, flags, rc);
    }
    for (size_t i = 0; i < sizeof bigs / sizeof *bigs; i++) big_free(bigs[i]);
    dyad_free(&lo);
    dyad_free(&hi);
    return out;
}

static PyObject *py_int_bounds(PyObject *self, PyObject *args) {
    PyObject *f64_layout, *lo_obj, *hi_obj, *ref_obj;
    long long itemsize;
    int is_signed, ovf;
    (void)self;
    if (!PyArg_ParseTuple(args, "LpiOOOO", &itemsize, &is_signed, &ovf, &f64_layout, &lo_obj, &hi_obj, &ref_obj)) return NULL;
    floatfmt f64;
    bigint f64_nan[2], lo, hi, ref, tol, low, high;
    bigint *bigs[] = {&f64_nan[0], &f64_nan[1], &lo, &hi, &ref, &tol, &low, &high};
    for (size_t i = 0; i < sizeof bigs / sizeof *bigs; i++) big_init(bigs[i]);
    PyObject *out = NULL;
    if (!layout_load(f64_layout, &f64, f64_nan) && !int_load(lo_obj, &lo) && !int_load(hi_obj, &hi) && !int_load(ref_obj, &ref)) {
        int rc = int_bounds(&tol, &low, &high, &f64, itemsize, is_signed, ovf, &lo, &hi, &ref);
        if (rc) {
            out = error(rc);
        } else {
            PyObject *a = int_store(&tol), *b = a ? int_store(&low) : NULL, *c = b ? int_store(&high) : NULL;
            out = c ? Py_BuildValue("(OOO)", a, b, c) : NULL;
            Py_XDECREF(a);
            Py_XDECREF(b);
            Py_XDECREF(c);
        }
    }
    for (size_t i = 0; i < sizeof bigs / sizeof *bigs; i++) big_free(bigs[i]);
    return out;
}

static PyObject *py_growth_terms(PyObject *self, PyObject *args) {
    PyObject *layout;
    int rnd, flush;
    long long depth, roundings, w;
    (void)self;
    if (!PyArg_ParseTuple(args, "OiLLLp", &layout, &rnd, &depth, &roundings, &w, &flush)) return NULL;
    floatfmt f;
    bigint nan[2];
    dyad gamma, constant, largest;
    big_init(&nan[0]);
    big_init(&nan[1]);
    dyad_init(&gamma);
    dyad_init(&constant);
    dyad_init(&largest);
    PyObject *out = NULL;
    if (!layout_load(layout, &f, nan)) {
        int bounded, rc = growth_terms(&gamma, &constant, &largest, &bounded, &f, rnd, depth, roundings, w, flush);
        if (rc) {
            out = error(rc);
        } else if (!bounded) {
            Py_INCREF(Py_None);
            out = Py_None;
        } else {
            PyObject *a = dyad_store(&gamma, 0), *b = a ? dyad_store(&constant, 0) : NULL, *c = b ? dyad_store(&largest, 0) : NULL;
            out = c ? PyTuple_Pack(3, a, b, c) : NULL;
            Py_XDECREF(a);
            Py_XDECREF(b);
            Py_XDECREF(c);
        }
    }
    big_free(&nan[0]);
    big_free(&nan[1]);
    dyad_free(&gamma);
    dyad_free(&constant);
    dyad_free(&largest);
    return out;
}

static PyObject *py_widest(PyObject *self, PyObject *args) {
    PyObject *layout, *center, *f64_layout, *found;
    int center_big, components;
    (void)self;
    if (!PyArg_ParseTuple(args, "OSpOOi", &layout, &center, &center_big, &f64_layout, &found, &components)) return NULL;
    Py_ssize_t n = PyList_Size(found);
    if (n < 0) return NULL;
    partfmt p, f64;
    bigint nan[2], f64_nan[2], tol;
    big_init(&nan[0]);
    big_init(&nan[1]);
    big_init(&f64_nan[0]);
    big_init(&f64_nan[1]);
    big_init(&tol);
    /* per result: tolerance, low and high (data, big-endian), and flags */
    const unsigned char **data = PyMem_Calloc(4 * (size_t)n + 1, sizeof *data);
    int *big = PyMem_Calloc(3 * (size_t)n + 1, sizeof *big), *part_flags = PyMem_Calloc((size_t)n + 1, sizeof *part_flags);
    value *vals = PyMem_Calloc(3 * (size_t)n + 4, sizeof *vals);
    PyObject *out = NULL, *tols = NULL, *lows = NULL, *highs = NULL, *flags = NULL;
    for (Py_ssize_t i = 0; vals && i < 3 * n + 4; i++) value_init(&vals[i]);
    char *s;
    Py_ssize_t len;
    if (!data || !big || !part_flags || !vals) {
        PyErr_NoMemory();
        goto done;
    }
    if (partfmt_load(layout, &p, nan) || partfmt_load(f64_layout, &f64, f64_nan) || PyBytes_AsStringAndSize(center, &s, &len)) goto done;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyList_GetItem(found, i), *a, *b, *c, *d;
        if (!item || !PyArg_ParseTuple(item, "SpSpSpS", &a, &big[3 * i], &b, &big[3 * i + 1], &c, &big[3 * i + 2], &d)) goto done;
        data[4 * i] = (const unsigned char *)PyBytes_AsString(a);
        data[4 * i + 1] = (const unsigned char *)PyBytes_AsString(b);
        data[4 * i + 2] = (const unsigned char *)PyBytes_AsString(c);
        data[4 * i + 3] = (const unsigned char *)PyBytes_AsString(d);
    }
    size_t size = part_size(&p), parts = (size_t)len / size;
    tols = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(8 * parts));
    lows = PyBytes_FromStringAndSize(NULL, len);
    highs = PyBytes_FromStringAndSize(NULL, len);
    flags = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(parts / (size_t)components));
    if (!tols || !lows || !highs || !flags) goto done;
    unsigned char *tol_out = (unsigned char *)PyBytes_AsString(tols), *low_out = (unsigned char *)PyBytes_AsString(lows);
    unsigned char *high_out = (unsigned char *)PyBytes_AsString(highs), *flag_out = (unsigned char *)PyBytes_AsString(flags);
    value *c = &vals[3 * n], *lo = &vals[3 * n + 1], *hi = &vals[3 * n + 2], *t = vals, *l = vals + n, *h = vals + 2 * n;
    int rc = 0;
    for (size_t k = 0; !rc && k < parts; k++) {
        if ((rc = part_read(c, &p, (const unsigned char *)s, k, center_big))) break;
        for (Py_ssize_t i = 0; !rc && i < n; i++) {
            int shift = components == 2 ? 3 * (int)(k & 1) : 0;
            part_flags[i] = data[4 * i + 3][k / (size_t)components] >> shift & FLAG_ALL;
            if (!(rc = part_read(&t[i], &f64, data[4 * i], k, big[3 * i])) && !(rc = part_read(&l[i], &p, data[4 * i + 1], k, big[3 * i + 1])))
                rc = part_read(&h[i], &p, data[4 * i + 2], k, big[3 * i + 2]);
        }
        int empty, part;
        if (rc || (rc = widest_part(&tol, lo, hi, &empty, &part, &f64.f, c, (size_t)n, t, l, h, part_flags))) break;
        bytes_store(tol_out + 8 * k, 8, &tol);
        if (empty) {
            lo->kind = hi->kind = VALUE_NAN; /* an empty range's ends are NaN */
            lo->neg = hi->neg = 0;
        }
        if ((rc = part_write(low_out, k, &p, lo)) || (rc = part_write(high_out, k, &p, hi))) break;
        if (components == 2 && k & 1) {
            flag_out[k / 2] = (unsigned char)(flag_out[k / 2] | part << 3);
        } else {
            flag_out[k / (size_t)components] = (unsigned char)part;
        }
    }
    if (rc) {
        error(rc);
        goto done;
    }
    out = PyTuple_Pack(4, tols, lows, highs, flags);
done:
    Py_XDECREF(tols);
    Py_XDECREF(lows);
    Py_XDECREF(highs);
    Py_XDECREF(flags);
    for (Py_ssize_t i = 0; vals && i < 3 * n + 4; i++) value_free(&vals[i]);
    PyMem_Free((void *)data);
    PyMem_Free(big);
    PyMem_Free(part_flags);
    PyMem_Free(vals);
    big_free(&nan[0]);
    big_free(&nan[1]);
    big_free(&f64_nan[0]);
    big_free(&f64_nan[1]);
    big_free(&tol);
    return out;
}

/* The parts of arrays as (bytes, big-endian) pairs crossing in calls: the data of each. */
static int data_load(PyObject *obj, const unsigned char **data, Py_ssize_t *len) {
    char *s;
    if (PyBytes_AsStringAndSize(obj, &s, len)) return -1;
    *data = (const unsigned char *)s;
    return 0;
}

/* An earlier result's values and radii for the attention kernels (Results as inputs): per element
   its value and its radius (widened by part_radius) as 128-bit little-endian integers at one
   exponent 2**e0 (signed, unsigned), and whether the radius bounds nothing. e0 is the finest unit of
   the values and radii where every |x| + r is below 2**125 of it; where it is not, the unit is coarser,
   each value is rounded down to it and its radius up, plus one unit where the value moved, so that
   [x' - r', x' + r'] holds [x - r, x + r] and |x'| + r' >= |x| + r. Two passes over the bytes
   (operands.c's rpack), the first for e0, so that nothing per element is held but the output; each in
   tiles of RPACK_CHUNK elements on the workers: radius_packed(...) makes the call, radius_packed_next
   gives each pass's tiles, radius_packed_tile runs one, radius_packed_result gives (values, radii, e0,
   unbounded). The second pass fills elements lo .. hi - 1 only, so that a part of the elements (the
   attention tolerance's head groups) has its output alone at the exponent of all of them; given e0 (of
   an earlier call's first pass), the first pass is left out. */
#define RPACK_CAPSULE "exactref._c.rpack"
#define RPACK_CHUNK ((size_t)1 << 18)

typedef struct {
    rpack x;
    size_t count, tiles;     /* the elements and the first pass's tiles (0 when e0 is given) */
    size_t lo, hi, tiles2;   /* the second pass's elements and its tiles */
    int phase;               /* 0 made, 1 the first pass's tiles, 2 the second's, 3 done */
    rpack_part_scan *parts;  /* per tile: its first pass */
    PyObject *keep;          /* the values' and radii's bytes, which the tiles read */
    PyObject *vv, *vr, *unb; /* the outputs, written by the tiles until the result hands them out */
} rpack_box;

static void rpack_box_free(rpack_box *b) {
    free(b->parts);
    Py_XDECREF(b->keep);
    Py_XDECREF(b->vv);
    Py_XDECREF(b->vr);
    Py_XDECREF(b->unb);
    free(b);
}

static void rpack_destroy(PyObject *capsule) {
    rpack_box *b = PyCapsule_GetPointer(capsule, RPACK_CAPSULE);
    if (b) rpack_box_free(b);
}

static PyObject *py_radius_packed(PyObject *self, PyObject *args) {
    PyObject *layout, *values, *radii, *emin_obj, *f64_layout, *e0_obj;
    int values_big, radii_big;
    Py_ssize_t count, lo, hi;
    (void)self;
    if (!PyArg_ParseTuple(args, "OSpOSpnOOnn", &layout, &values, &values_big, &f64_layout, &radii, &radii_big, &count, &emin_obj, &e0_obj, &lo, &hi))
        return NULL;
    if (lo < 0 || hi < lo || hi > count) {
        PyErr_SetString(PyExc_ValueError, "radius_packed: lo and hi outside the elements");
        return NULL;
    }
    rpack_box *b = calloc(1, sizeof *b);
    if (!b) return PyErr_NoMemory();
    rpack_init(&b->x);
    bigint nan[4];
    for (int k = 0; k < 4; k++) big_init(&nan[k]);
    PyObject *out = NULL;
    const unsigned char *vd, *rd;
    Py_ssize_t vlen, rlen;
    size_t m = count > 0 ? (size_t)count : 0, part = (size_t)(hi - lo);
    b->count = m;
    b->tiles = e0_obj == Py_None ? (m + RPACK_CHUNK - 1) / RPACK_CHUNK : 0;
    b->lo = (size_t)lo;
    b->hi = (size_t)hi;
    b->tiles2 = (part + RPACK_CHUNK - 1) / RPACK_CHUNK;
    b->x.base = b->lo;
    if ((emin_obj != Py_None && exp_load(emin_obj, &b->x.emin)) || (e0_obj != Py_None && exp_load(e0_obj, &b->x.e0)) || partfmt_load(layout, &b->x.p, nan) ||
        partfmt_load(f64_layout, &b->x.f64, nan + 2) || data_load(values, &vd, &vlen) || data_load(radii, &rd, &rlen))
        goto done;
    /* only decoded: their NaN encodings not kept */
    b->x.p.f.nan[0] = b->x.p.f.nan[1] = b->x.f64.f.nan[0] = b->x.f64.f.nan[1] = NULL;
    if (part_size(&b->x.p) > 8 || part_size(&b->x.f64) > 8) {
        PyErr_SetString(PyExc_ValueError, "radius_packed: formats of at most 64 bits");
        goto done;
    }
    b->x.vsize = part_size(&b->x.p);
    b->x.rsize = part_size(&b->x.f64);
    b->x.values = vd;
    b->x.radii = rd;
    b->x.values_big = values_big;
    b->x.radii_big = radii_big;
    b->x.has_emin = emin_obj != Py_None;
    if (!(b->vv = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(16 * part))) || !(b->vr = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(16 * part))) ||
        !(b->unb = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)part)) || !(b->keep = PyTuple_Pack(2, values, radii)))
        goto done;
    if (!(b->parts = calloc(b->tiles + 1, sizeof *b->parts))) {
        PyErr_NoMemory();
        goto done;
    }
    /* new objects: ours to fill until the result hands them out */
    b->x.ov = (unsigned char *)PyBytes_AsString(b->vv);
    b->x.orad = (unsigned char *)PyBytes_AsString(b->vr);
    b->x.ou = (unsigned char *)PyBytes_AsString(b->unb);
    if ((out = PyCapsule_New(b, RPACK_CAPSULE, rpack_destroy))) b = NULL;
done:
    for (int k = 0; k < 4; k++) big_free(&nan[k]);
    if (b) rpack_box_free(b);
    return out;
}

/* radius_packed_next(call): the tiles of the next pass, 0 when done */
static PyObject *py_radius_packed_next(PyObject *self, PyObject *args) {
    PyObject *capsule;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    rpack_box *b = PyCapsule_GetPointer(capsule, RPACK_CAPSULE);
    if (!b) return NULL;
    if (b->phase == 0) {
        b->phase = 1;
        if (b->tiles) return PyLong_FromSize_t(b->tiles);
    }
    if (b->phase == 1) {
        if (b->tiles) rpack_fold(&b->x, b->parts, b->tiles); /* else e0 was given */
        b->phase = 2;
        if (b->tiles2) return PyLong_FromSize_t(b->tiles2);
    }
    b->phase = 3;
    return PyLong_FromLong(0);
}

/* radius_packed_tile(call, t): runs tile t of the current pass */
static PyObject *py_radius_packed_tile(PyObject *self, PyObject *args) {
    PyObject *capsule;
    Py_ssize_t t;
    (void)self;
    if (!PyArg_ParseTuple(args, "On", &capsule, &t)) return NULL;
    rpack_box *b = PyCapsule_GetPointer(capsule, RPACK_CAPSULE);
    if (!b) return NULL;
    int first = b->phase == 1;
    if ((b->phase != 1 && b->phase != 2) || t < 0 || (size_t)t >= (first ? b->tiles : b->tiles2)) {
        PyErr_SetString(PyExc_ValueError, "radius_packed_tile: no such tile");
        return NULL;
    }
    size_t start = first ? 0 : b->lo, end = first ? b->count : b->hi;
    size_t i0 = start + (size_t)t * RPACK_CHUNK, i1 = end - i0 > RPACK_CHUNK ? i0 + RPACK_CHUNK : end;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = first ? rpack_scan(&b->parts[t], &b->x, i0, i1) : rpack_fill(&b->x, i0, i1);
    Py_END_ALLOW_THREADS
    if (rc) return error(rc);
    return Py_NewRef(Py_None);
}

/* radius_packed_result(call): (values, radii, e0, unbounded) once both passes are done */
static PyObject *py_radius_packed_result(PyObject *self, PyObject *args) {
    PyObject *capsule;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    rpack_box *b = PyCapsule_GetPointer(capsule, RPACK_CAPSULE);
    if (!b) return NULL;
    if (b->phase != 3) {
        PyErr_SetString(PyExc_ValueError, "radius_packed_result: the passes are not done");
        return NULL;
    }
    return Py_BuildValue("(OOLO)", b->vv, b->vr, (long long)b->x.e0, b->unb);
}

/* Per part, f(the parts' values, flags) for marks and covers: a list of ints. */
static PyObject *py_part_checks(PyObject *args, int covers) {
    PyObject *layout, *f64_layout, *a, *lo, *hi_or_tol, *flags;
    int a_big, lo_big, third_big, components;
    if (!PyArg_ParseTuple(args, "OOSpSpSpSi", &layout, &f64_layout, &a, &a_big, &lo, &lo_big, &hi_or_tol, &third_big, &flags, &components)) return NULL;
    partfmt p, f64;
    bigint nan[2], f64_nan[2];
    value x, l, t;
    big_init(&nan[0]);
    big_init(&nan[1]);
    big_init(&f64_nan[0]);
    big_init(&f64_nan[1]);
    value_init(&x);
    value_init(&l);
    value_init(&t);
    PyObject *out = NULL;
    const unsigned char *ad, *ld, *td, *fd;
    Py_ssize_t alen, llen, tlen, flen;
    if (!partfmt_load(layout, &p, nan) && !partfmt_load(f64_layout, &f64, f64_nan) && !data_load(a, &ad, &alen) && !data_load(lo, &ld, &llen) &&
        !data_load(hi_or_tol, &td, &tlen) && !data_load(flags, &fd, &flen) && (out = PyList_New(flen * components))) {
        int rc = 0;
        for (Py_ssize_t k = 0; !rc && k < flen * components; k++) {
            int f = fd[k / components] >> (components == 2 ? 3 * (int)(k & 1) : 0) & FLAG_ALL;
            if ((rc = part_read(&x, &p, ad, (size_t)k, a_big)) || (rc = part_read(&l, &p, ld, (size_t)k, lo_big)) ||
                (rc = part_read(&t, covers ? &p : &f64, td, (size_t)k, third_big)))
                break;
            PyList_SetItem(out, k, PyLong_FromLong(covers ? part_covers(&x, &l, &t, f) : part_marks(f, &x, &l, &t)));
        }
        if (rc) {
            Py_CLEAR(out);
            error(rc);
        }
    }
    big_free(&nan[0]);
    big_free(&nan[1]);
    big_free(&f64_nan[0]);
    big_free(&f64_nan[1]);
    value_free(&x);
    value_free(&l);
    value_free(&t);
    return out;
}

static PyObject *py_marks(PyObject *self, PyObject *args) {
    (void)self;
    return py_part_checks(args, 0);
}

static PyObject *py_covers(PyObject *self, PyObject *args) {
    (void)self;
    return py_part_checks(args, 1);
}

static PyObject *py_recode(PyObject *self, PyObject *args) {
    PyObject *data, *from_layout, *to_layout;
    int big_endian;
    (void)self;
    if (!PyArg_ParseTuple(args, "SpOO", &data, &big_endian, &from_layout, &to_layout)) return NULL;
    partfmt from, to;
    bigint from_nan[2], to_nan[2];
    value x;
    big_init(&from_nan[0]);
    big_init(&from_nan[1]);
    big_init(&to_nan[0]);
    big_init(&to_nan[1]);
    value_init(&x);
    PyObject *out = NULL;
    const unsigned char *src;
    Py_ssize_t len;
    if (!partfmt_load(from_layout, &from, from_nan) && !partfmt_load(to_layout, &to, to_nan) && !data_load(data, &src, &len)) {
        size_t from_size = part_size(&from);
        size_t to_size = part_size(&to), parts = (size_t)len / from_size;
        if ((out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(parts * to_size)))) {
            unsigned char *dst = (unsigned char *)PyBytes_AsString(out); /* a new object: still ours to fill */
            int rc = 0;
            for (size_t k = 0; !rc && k < parts; k++)
                if (!(rc = part_read(&x, &from, src, k, big_endian))) rc = part_write(dst, k, &to, &x);
            if (rc) {
                Py_CLEAR(out);
                error(rc);
            }
        }
    }
    big_free(&from_nan[0]);
    big_free(&from_nan[1]);
    big_free(&to_nan[0]);
    big_free(&to_nan[1]);
    value_free(&x);
    return out;
}

/* Whether a C double is binary64 stored as a uint64_t holding its encoding: then memcpy makes a float64 encoding its
   double, the float struct.unpack("<d") reads from it (FloatValue.__float__). */
static int binary64_doubles(void) {
    static const double probe = 9006104071832581.0; /* encoded 0x433FFF0102030405: eight distinct bytes */
    uint64_t w;
    if (sizeof probe != sizeof w) return 0;
    memcpy(&w, &probe, sizeof w);
    return w == UINT64_C(0x433FFF0102030405);
}

/* An array's elements as Python scalars (Array.scalars): bools from bytes (layout None), ints, and floats or complex
   numbers (components 2) by their float64 encodings (part_float64); NotImplemented for floats where a double is not
   binary64 (binary64_doubles), whose elements the per-element codec reads. */
static PyObject *py_scalars(PyObject *self, PyObject *args) {
    PyObject *data, *layout;
    int big_endian, components;
    (void)self;
    if (!PyArg_ParseTuple(args, "SpOi", &data, &big_endian, &layout, &components)) return NULL;
    const unsigned char *src;
    Py_ssize_t len;
    if (data_load(data, &src, &len)) return NULL;
    PyObject *out = NULL;
    if (layout == Py_None) {
        if ((out = PyList_New(len)))
            for (Py_ssize_t i = 0; i < len; i++) PyList_SetItem(out, i, PyBool_FromLong(src[i]));
        return out;
    }
    partfmt p, f64;
    bigint nan[2];
    big_init(&nan[0]);
    big_init(&nan[1]);
    int held = 0, rc = 0;
    if (!partfmt_load(layout, &p, nan)) {
        if (!p.is_int) {
            f64.is_int = 0;
            rc = floatfmt_set(&f64.f, 11, 52, SPECIAL_IEEE, NULL, 1);
            if (!rc) rc = format_holds(&held, &f64, &p);
        }
        size_t n = (size_t)len / part_size(&p) / (size_t)components;
        if (rc) {
            error(rc);
        } else if (!p.is_int && !binary64_doubles()) {
            out = Py_NewRef(Py_NotImplemented);
        } else if ((out = PyList_New((Py_ssize_t)n))) {
            for (size_t i = 0; i < n; i++) {
                PyObject *item = NULL;
                if (p.is_int) {
                    int kind, neg;
                    uint64_t sig;
                    int64_t e;
                    element_decode(&kind, &neg, &sig, &e, &p, part_bits(&p, src, i, big_endian));
                    /* a negative value -sig as -(sig - 1) - 1, which stays in range down to -2**63 */
                    item = neg ? PyLong_FromLongLong(-(long long)(sig - 1) - 1) : PyLong_FromUnsignedLongLong(sig);
                } else {
                    uint64_t w[2] = {0, 0};
                    double d[2];
                    for (int c = 0; !rc && c < components; c++)
                        rc = part_float64(&w[c], &p.f, &f64.f, held, part_bits(&p, src, (size_t)components * i + (size_t)c, big_endian));
                    memcpy(d, w, sizeof d);
                    if (!rc) item = components == 2 ? PyComplex_FromDoubles(d[0], d[1]) : PyFloat_FromDouble(d[0]);
                }
                if (!item) {
                    Py_CLEAR(out);
                    if (rc) error(rc);
                    break;
                }
                PyList_SetItem(out, (Py_ssize_t)i, item);
            }
        }
    }
    big_free(&nan[0]);
    big_free(&nan[1]);
    return out;
}

/* Whether every element (every) or some element of an array is nonzero by its encoding (Array.all and Array.any):
   elements_nonzero. */
static PyObject *py_nonzero(PyObject *self, PyObject *args) {
    PyObject *data, *layout;
    int big_endian, components, every;
    (void)self;
    if (!PyArg_ParseTuple(args, "SpOip", &data, &big_endian, &layout, &components, &every)) return NULL;
    partfmt p;
    bigint nan[2];
    big_init(&nan[0]);
    big_init(&nan[1]);
    PyObject *out = NULL;
    const unsigned char *src;
    Py_ssize_t len;
    if (!partfmt_load(layout, &p, nan) && !data_load(data, &src, &len))
        out = PyBool_FromLong(elements_nonzero(&p, src, (size_t)len / part_size(&p) / (size_t)components, (size_t)components, big_endian, every));
    big_free(&nan[0]);
    big_free(&nan[1]);
    return out;
}

static PyObject *py_holds(PyObject *self, PyObject *args) {
    PyObject *f_obj, *g_obj;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &f_obj, &g_obj)) return NULL;
    partfmt f, g;
    bigint f_nan[2], g_nan[2];
    big_init(&f_nan[0]);
    big_init(&f_nan[1]);
    big_init(&g_nan[0]);
    big_init(&g_nan[1]);
    PyObject *out = NULL;
    if (!partfmt_load(f_obj, &f, f_nan) && !partfmt_load(g_obj, &g, g_nan)) {
        int holds, rc = format_holds(&holds, &f, &g);
        out = rc ? error(rc) : PyBool_FromLong(holds);
    }
    big_free(&f_nan[0]);
    big_free(&f_nan[1]);
    big_free(&g_nan[0]);
    big_free(&g_nan[1]);
    return out;
}

static PyObject *py_binary_holding(PyObject *self, PyObject *args) {
    long long bits, low, high;
    (void)self;
    if (!PyArg_ParseTuple(args, "LLL", &bits, &low, &high)) return NULL;
    int64_t e, m;
    binary_holding(&e, &m, bits, low, high);
    return Py_BuildValue("(LL)", (long long)e, (long long)m);
}

static PyObject *py_radius_parts(PyObject *self, PyObject *args) {
    PyObject *layout, *values, *radii, *emin_obj, *f64_layout;
    int values_big, radii_big;
    Py_ssize_t start, count;
    (void)self;
    if (!PyArg_ParseTuple(args, "OSpOSpnnO", &layout, &values, &values_big, &f64_layout, &radii, &radii_big, &start, &count, &emin_obj)) return NULL;
    partfmt p, f64;
    bigint nan[2], f64_nan[2], bits;
    value x, r;
    big_init(&nan[0]);
    big_init(&nan[1]);
    big_init(&f64_nan[0]);
    big_init(&f64_nan[1]);
    big_init(&bits);
    value_init(&x);
    value_init(&r);
    PyObject *out = NULL;
    const unsigned char *vd, *rd;
    Py_ssize_t vlen, rlen;
    int64_t emin = 0;
    if ((emin_obj == Py_None || !exp_load(emin_obj, &emin)) && !partfmt_load(layout, &p, nan) && !partfmt_load(f64_layout, &f64, f64_nan) &&
        !data_load(values, &vd, &vlen) && !data_load(radii, &rd, &rlen) && (out = PyBytes_FromStringAndSize(NULL, 8 * count))) {
        unsigned char *dst = (unsigned char *)PyBytes_AsString(out); /* a new object: still ours to fill */
        int rc = 0;
        for (Py_ssize_t i = 0; !rc && i < count; i++) {
            size_t k = (size_t)(start + i);
            bigint a;
            if ((rc = part_read(&r, &f64, rd, k, radii_big)) || (rc = part_read(&x, &p, vd, k, values_big)) ||
                (rc = part_radius(&r, &x, emin_obj != Py_None ? &emin : NULL)))
                break;
            /* the radius as a float64, rounded up: exact unless a wider format's |x| has more bits */
            a = r.v.m;
            a.neg = 0;
            if (!(rc = floatfmt_encode(&bits, &f64.f, r.kind, r.neg, &a, r.v.e, ROUND_TOWARD_POSITIVE, OVERFLOW_IEEE))) bytes_store(dst + 8 * i, 8, &bits);
        }
        if (rc) {
            Py_CLEAR(out);
            error(rc);
        }
    }
    big_free(&nan[0]);
    big_free(&nan[1]);
    big_free(&f64_nan[0]);
    big_free(&f64_nan[1]);
    big_free(&bits);
    value_free(&x);
    value_free(&r);
    return out;
}

static PyObject *py_read_operand(PyObject *self, PyObject *args) {
    PyObject *data, *layout;
    int big_endian;
    Py_ssize_t rows, cols, row_step, col_step;
    long long main_bits;
    (void)self;
    if (!PyArg_ParseTuple(args, "SpOnnnnL", &data, &big_endian, &layout, &rows, &cols, &row_step, &col_step, &main_bits)) return NULL;
    partfmt f;
    bigint nan[2];
    big_init(&nan[0]);
    big_init(&nan[1]);
    operand x;
    operand_init(&x);
    PyObject *out = NULL;
    const unsigned char *src;
    Py_ssize_t len;
    if (!partfmt_load(layout, &f, nan) && !data_load(data, &src, &len)) {
        int rc = read_operand(&x, &f, src, big_endian, (size_t)rows, (size_t)cols, (size_t)row_step, (size_t)col_step, rows ? (size_t)rows : 1, 0,
                              main_bits);
        size_t n = (size_t)rows * (size_t)cols, words = x.slots_hi ? 2 : 1;
        PyObject *slots = rc ? NULL : PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(8 * words * n));
        if (slots) {
            unsigned char *s = (unsigned char *)PyBytes_AsString(slots); /* a new object: still ours to fill */
            for (size_t i = 0; i < n; i++)
                for (size_t w = 0; w < words; w++)
                    for (int k = 0; k < 8; k++) s[8 * (words * i + w) + (size_t)k] = (unsigned char)((w ? x.slots_hi : x.slots)[i] >> (8 * k));
            PyObject *flagged = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(8 * (n / 64 + 1)));
            if (flagged) {
                unsigned char *b = (unsigned char *)PyBytes_AsString(flagged);
                for (size_t i = 0; i < n / 64 + 1; i++)
                    for (int k = 0; k < 8; k++) b[8 * i + (size_t)k] = (unsigned char)(x.flagged[i] >> (8 * k));
                /* the summaries as bytes objects: '#' formats differ across the stable ABI's versions */
                PyObject *summary = PyBytes_FromStringAndSize((const char *)x.summary, (Py_ssize_t)(x.rows * x.blocks));
                PyObject *special = summary ? PyBytes_FromStringAndSize((const char *)x.special_rows, rows) : NULL;
                if (special) {
                    /* "N" takes the references, also on failure */
                    out = Py_BuildValue("(LLNNNNn)", (long long)x.exponent, (long long)x.width, slots, flagged, summary, special,
                                        (Py_ssize_t)x.flagged_count);
                } else {
                    Py_XDECREF(summary);
                    Py_DECREF(flagged);
                    Py_DECREF(slots);
                }
                slots = NULL;
            }
            Py_XDECREF(slots);
        } else if (rc) {
            error(rc);
        }
    }
    operand_free(&x);
    big_free(&nan[0]);
    big_free(&nan[1]);
    return out;
}

/* The product kernel's operands (capsules) and calls. */
#define OPERAND_CAPSULE "exactref._c.operand"

static void operand_destroy(PyObject *capsule) {
    operand *x = PyCapsule_GetPointer(capsule, OPERAND_CAPSULE);
    if (x) {
        operand_free(x);
        free(x);
    }
}

static const operand *operand_get(PyObject *capsule) { return PyCapsule_GetPointer(capsule, OPERAND_CAPSULE); }

static PyObject *py_operand(PyObject *self, PyObject *args) {
    PyObject *data, *layout, *given = Py_None;
    int big_endian;
    Py_ssize_t start, rows, cols, row_step, col_step, inner, outer_step, terms;
    int wide, given_flags = 0;
    long long given_top = 0;
    (void)self;
    if (!PyArg_ParseTuple(args, "SpOnnnnnnnnp|O", &data, &big_endian, &layout, &start, &rows, &cols, &row_step, &col_step, &inner, &outer_step, &terms,
                          &wide, &given))
        return NULL;
    /* a part of a larger operand: that operand's (top, flagged) from operand_scan, so each slot is read as in the whole */
    if (given != Py_None && !PyArg_ParseTuple(given, "Lp", &given_top, &given_flags)) return NULL;
    partfmt f;
    bigint nan[2];
    big_init(&nan[0]);
    big_init(&nan[1]);
    PyObject *out = NULL;
    const unsigned char *src;
    Py_ssize_t len;
    if (!partfmt_load(layout, &f, nan) && !data_load(data, &src, &len)) {
        operand *x = malloc(sizeof *x);
        int rc = EXACTREF_NOMEM;
        if (x) {
            size_t itemsize = part_size(&f);
            int64_t bits = product_main_bits((size_t)terms, wide);
            int64_t top = (int64_t)given_top;
            Py_BEGIN_ALLOW_THREADS
            rc = read_operand_given(x, &f, src + (size_t)start * itemsize, big_endian, (size_t)rows, (size_t)cols, (size_t)row_step, (size_t)col_step,
                                    inner > 0 ? (size_t)inner : 1, (size_t)outer_step, bits, given == Py_None ? NULL : &top);
            Py_END_ALLOW_THREADS
            if (!rc && given_flags && !x->flagged_count) x->flagged_count = 1; /* the whole operand flags a slot: its users read flagged_count as that */
        }
        if (rc) {
            free(x);
            error(rc);
        } else if (!(out = PyCapsule_New(x, OPERAND_CAPSULE, operand_destroy))) {
            operand_free(x);
            free(x);
        }
    }
    big_free(&nan[0]);
    big_free(&nan[1]);
    return out;
}

/* A read on the workers (_native.operand): operand_rows_top(data, big_endian, layout, start, cols, row_step, col_step, inner,
   outer_step, r0, r1) gives rows r0 .. r1 - 1's (any, top); operand_new(layout, rows, cols, terms, wide, any, top) the operand
   at the main exponent of the largest top; operand_fill(x, data, big_endian, layout, start, row_step, col_step, inner,
   outer_step, r0, r1) reads rows r0 .. r1 - 1 into it, giving their (width, flagged); operand_done(x, width, flagged,
   given_flags) sets the totals. Ranges whose first slot is a multiple of 64 fill at once (operand_fill). */
static PyObject *py_operand_rows_top(PyObject *self, PyObject *args) {
    PyObject *data, *layout;
    int big_endian;
    Py_ssize_t start, cols, row_step, col_step, inner, outer_step, r0, r1;
    (void)self;
    if (!PyArg_ParseTuple(args, "SpOnnnnnnnn", &data, &big_endian, &layout, &start, &cols, &row_step, &col_step, &inner, &outer_step, &r0, &r1)) return NULL;
    partfmt f;
    bigint nan[2];
    big_init(&nan[0]);
    big_init(&nan[1]);
    PyObject *out = NULL;
    const unsigned char *src;
    Py_ssize_t len;
    if (!partfmt_load(layout, &f, nan) && !data_load(data, &src, &len)) {
        int any;
        int64_t top;
        Py_BEGIN_ALLOW_THREADS
        operand_rows_top(&any, &top, &f, src + (size_t)start * part_size(&f), big_endian, (size_t)cols, (size_t)row_step, (size_t)col_step,
                         inner > 0 ? (size_t)inner : 1, (size_t)outer_step, (size_t)r0, (size_t)r1);
        Py_END_ALLOW_THREADS
        out = Py_BuildValue("(NL)", PyBool_FromLong(any), (long long)top);
    }
    big_free(&nan[0]);
    big_free(&nan[1]);
    return out;
}

static PyObject *py_operand_new(PyObject *self, PyObject *args) {
    PyObject *layout;
    Py_ssize_t rows, cols, terms;
    int wide, any;
    long long top;
    (void)self;
    if (!PyArg_ParseTuple(args, "OnnnppL", &layout, &rows, &cols, &terms, &wide, &any, &top)) return NULL;
    partfmt f;
    bigint nan[2];
    big_init(&nan[0]);
    big_init(&nan[1]);
    PyObject *out = NULL;
    if (!partfmt_load(layout, &f, nan)) {
        operand *x = malloc(sizeof *x);
        int rc = x ? operand_alloc(x, &f, (size_t)rows, (size_t)cols, product_main_bits((size_t)terms, wide), any, (int64_t)top) : EXACTREF_NOMEM;
        if (rc) {
            free(x);
            error(rc);
        } else if (!(out = PyCapsule_New(x, OPERAND_CAPSULE, operand_destroy))) {
            operand_free(x);
            free(x);
        }
    }
    big_free(&nan[0]);
    big_free(&nan[1]);
    return out;
}

static PyObject *py_operand_fill(PyObject *self, PyObject *args) {
    PyObject *capsule, *data, *layout;
    int big_endian;
    Py_ssize_t start, row_step, col_step, inner, outer_step, r0, r1;
    (void)self;
    if (!PyArg_ParseTuple(args, "OSpOnnnnnnn", &capsule, &data, &big_endian, &layout, &start, &row_step, &col_step, &inner, &outer_step, &r0, &r1)) return NULL;
    operand *x = PyCapsule_GetPointer(capsule, OPERAND_CAPSULE);
    if (!x) return NULL;
    if (r0 < 0 || r1 < r0 || (size_t)r1 > x->rows) {
        PyErr_SetString(PyExc_ValueError, "operand_fill: rows outside the operand");
        return NULL;
    }
    partfmt f;
    bigint nan[2];
    big_init(&nan[0]);
    big_init(&nan[1]);
    PyObject *out = NULL;
    const unsigned char *src;
    Py_ssize_t len;
    if (!partfmt_load(layout, &f, nan) && !data_load(data, &src, &len)) {
        int64_t width;
        size_t flagged;
        Py_BEGIN_ALLOW_THREADS
        operand_fill(x, &f, src + (size_t)start * part_size(&f), big_endian, (size_t)row_step, (size_t)col_step, inner > 0 ? (size_t)inner : 1,
                     (size_t)outer_step, (size_t)r0, (size_t)r1, &width, &flagged);
        Py_END_ALLOW_THREADS
        out = Py_BuildValue("(Ln)", (long long)width, (Py_ssize_t)flagged);
    }
    big_free(&nan[0]);
    big_free(&nan[1]);
    return out;
}

static PyObject *py_operand_done(PyObject *self, PyObject *args) {
    PyObject *capsule;
    long long width;
    Py_ssize_t flagged;
    int given_flags;
    (void)self;
    if (!PyArg_ParseTuple(args, "OLnp", &capsule, &width, &flagged, &given_flags)) return NULL;
    operand *x = PyCapsule_GetPointer(capsule, OPERAND_CAPSULE);
    if (!x) return NULL;
    x->width = (int64_t)width;
    x->flagged_count = (size_t)flagged;
    if (given_flags && !x->flagged_count) x->flagged_count = 1; /* as py_operand: the whole operand flags a slot */
    return Py_NewRef(Py_None);
}

/* operand_scan(data, big_endian, layout, start, rows, cols, row_step, col_step, inner, outer_step, terms, wide): an operand's
   values without reading it (scan_operand): (top, low, any, nonfinite, the main bits a read for terms takes) */
static PyObject *py_operand_scan(PyObject *self, PyObject *args) {
    PyObject *data, *layout;
    int big_endian, wide;
    Py_ssize_t start, rows, cols, row_step, col_step, inner, outer_step, terms;
    (void)self;
    if (!PyArg_ParseTuple(args, "SpOnnnnnnnnp", &data, &big_endian, &layout, &start, &rows, &cols, &row_step, &col_step, &inner, &outer_step, &terms,
                          &wide))
        return NULL;
    partfmt f;
    bigint nan[2];
    big_init(&nan[0]);
    big_init(&nan[1]);
    PyObject *out = NULL;
    const unsigned char *src;
    Py_ssize_t len;
    if (!partfmt_load(layout, &f, nan) && !data_load(data, &src, &len)) {
        int64_t top, low, bits = product_main_bits((size_t)terms, wide);
        int any, nonfinite;
        size_t itemsize = part_size(&f);
        Py_BEGIN_ALLOW_THREADS
        scan_operand(&top, &low, &any, &nonfinite, &f, src + (size_t)start * itemsize, big_endian, (size_t)rows, (size_t)cols, (size_t)row_step,
                     (size_t)col_step, inner > 0 ? (size_t)inner : 1, (size_t)outer_step);
        Py_END_ALLOW_THREADS
        out = Py_BuildValue("(LLNNL)", (long long)top, (long long)low, PyBool_FromLong(any), PyBool_FromLong(nonfinite), (long long)bits);
    }
    big_free(&nan[0]);
    big_free(&nan[1]);
    return out;
}

/* An operand's batches as a count c (batch t reads the operand's batch t mod c) or bytes of
   native int64 (batch t reads row bases[t] on). */
static int bases_load(const int64_t **bases, size_t *count, PyObject *obj) {
    *bases = NULL;
    *count = 1;
    if (PyLong_Check(obj)) {
        Py_ssize_t c = PyLong_AsSsize_t(obj);
        if (c < 0 && PyErr_Occurred()) return -1;
        *count = c > 0 ? (size_t)c : 1;
        return 0;
    }
    const unsigned char *data;
    Py_ssize_t len;
    if (data_load(obj, &data, &len)) return -1;
    *bases = (const int64_t *)(const void *)data;
    return 0;
}

static int addressing_load(addressing *x, PyObject *a_bases, PyObject *b_bases, Py_ssize_t m, Py_ssize_t n) {
    x->m = (size_t)m;
    x->n = (size_t)n;
    return bases_load(&x->bases[0], &x->counts[0], a_bases) || bases_load(&x->bases[1], &x->counts[1], b_bases) ? -1 : 0;
}

static PyObject *py_operand_parts(PyObject *self, PyObject *args) {
    PyObject *b_obj;
    int coefficients;
    (void)self;
    if (!PyArg_ParseTuple(args, "Op", &b_obj, &coefficients)) return NULL;
    const operand *b = operand_get(b_obj);
    if (!b) return NULL;
    operand *re = malloc(sizeof *re), *im = malloc(sizeof *im);
    int rc = re && im ? operand_parts(re, im, b, coefficients) : EXACTREF_NOMEM;
    if (rc) {
        free(re);
        free(im);
        return error(rc);
    }
    PyObject *x = PyCapsule_New(re, OPERAND_CAPSULE, operand_destroy), *y = x ? PyCapsule_New(im, OPERAND_CAPSULE, operand_destroy) : NULL;
    if (!x) {
        operand_free(re);
        free(re);
    }
    if (!y) {
        operand_free(im);
        free(im);
        Py_XDECREF(x);
        return NULL;
    }
    PyObject *out = Py_BuildValue("(NN)", x, y);
    return out;
}

/* The elements of x and y in turn, each width bytes: x0 y0 x1 y1 ... */
static PyObject *py_interleave(PyObject *self, PyObject *args) {
    PyObject *x_obj, *y_obj;
    Py_ssize_t width;
    (void)self;
    if (!PyArg_ParseTuple(args, "SSn", &x_obj, &y_obj, &width)) return NULL;
    const unsigned char *x, *y;
    Py_ssize_t nx, ny;
    if (data_load(x_obj, &x, &nx) || data_load(y_obj, &y, &ny)) return NULL;
    PyObject *out = PyBytes_FromStringAndSize(NULL, nx + ny);
    if (!out) return NULL;
    unsigned char *o = (unsigned char *)PyBytes_AsString(out);
    size_t w = (size_t)width, n = (size_t)nx / w;
    for (size_t i = 0; i < n; i++) {
        memcpy(o + 2 * i * w, x + i * w, w);
        memcpy(o + (2 * i + 1) * w, y + i * w, w);
    }
    return out;
}

/* A complex output's flags from its parts': the real part's, and the imaginary part's shifted by
   3 (result.py's IMAG). */
static PyObject *py_pair_flags(PyObject *self, PyObject *args) {
    PyObject *x_obj, *y_obj;
    (void)self;
    if (!PyArg_ParseTuple(args, "SS", &x_obj, &y_obj)) return NULL;
    const unsigned char *x, *y;
    Py_ssize_t n, ny;
    if (data_load(x_obj, &x, &n) || data_load(y_obj, &y, &ny)) return NULL;
    PyObject *out = PyBytes_FromStringAndSize(NULL, n);
    if (!out) return NULL;
    unsigned char *o = (unsigned char *)PyBytes_AsString(out);
    for (Py_ssize_t i = 0; i < n; i++) o[i] = (unsigned char)(x[i] | y[i] << 3);
    return out;
}

/* A transposing gather copies its last two axes in tiles of GATHER_TILE x GATHER_TILE elements,
   so each source line it reads serves a whole tile row. */
#define GATHER_TILE 16

/* n elements of w bytes, step bytes apart from src, to consecutive places at dst. Each case copies
   a constant size, a load and a store, where a memcpy of a run-time size is a call. */
static void gather_elements(unsigned char *dst, const unsigned char *src, long long step, size_t n, size_t w) {
    switch (w) {
    case 1: for (size_t j = 0; j < n; j++) memcpy(dst + j, src + (long long)j * step, 1); break;
    case 2: for (size_t j = 0; j < n; j++) memcpy(dst + 2 * j, src + (long long)j * step, 2); break;
    case 4: for (size_t j = 0; j < n; j++) memcpy(dst + 4 * j, src + (long long)j * step, 4); break;
    case 8: for (size_t j = 0; j < n; j++) memcpy(dst + 8 * j, src + (long long)j * step, 8); break;
    case 16: for (size_t j = 0; j < n; j++) memcpy(dst + 16 * j, src + (long long)j * step, 16); break;
    default: for (size_t j = 0; j < n; j++) memcpy(dst + w * j, src + (long long)j * step, w); break;
    }
}

/* rows x cols elements of w bytes, row_step and col_step bytes apart from src, to dst in row-major
   order, a tile at a time. */
static void gather_tiles(unsigned char *dst, const unsigned char *src, long long row_step, long long col_step, size_t rows, size_t cols,
                         size_t w) {
    for (size_t i0 = 0; i0 < rows; i0 += GATHER_TILE)
        for (size_t j0 = 0; j0 < cols; j0 += GATHER_TILE) {
            size_t i1 = rows - i0 < GATHER_TILE ? rows : i0 + GATHER_TILE, n = cols - j0 < GATHER_TILE ? cols - j0 : GATHER_TILE;
            for (size_t i = i0; i < i1; i++)
                gather_elements(dst + (i * cols + j0) * w, src + ((long long)i * row_step + (long long)j0 * col_step), col_step, n, w);
        }
}

/* A DLPack tensor's elements in row-major order (_dlpack.py): w-byte elements from the address,
   axes of the given sizes and steps in elements (negative and 0 too). Neighbouring axes merge
   where the outer one steps over the inner one whole, so a contiguous tensor is one memcpy and a
   slice one per row. A strided last axis is copied element by element, in tiles of the last two
   axes when the axis before it steps less. Offsets stay integers; only existing elements'
   addresses are formed. */
static PyObject *py_gather(PyObject *self, PyObject *args) {
    PyObject *address, *shape_obj, *strides_obj;
    Py_ssize_t width;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOn", &address, &shape_obj, &strides_obj, &width)) return NULL;
    const unsigned char *base = PyLong_AsVoidPtr(address);
    Py_ssize_t ndim = PyTuple_Size(shape_obj);
    if (PyErr_Occurred()) return NULL;
    long long *sh = PyMem_Malloc(3 * (size_t)(ndim + 1) * sizeof *sh), *st = sh + ndim + 1, *idx = st + ndim + 1;
    if (!sh) return PyErr_NoMemory();
    size_t w = (size_t)width, total = w;
    Py_ssize_t k = 0;
    for (Py_ssize_t a = 0; a < ndim; a++) {
        long long n = PyLong_AsLongLong(PyTuple_GetItem(shape_obj, a)), s = PyLong_AsLongLong(PyTuple_GetItem(strides_obj, a)) * (long long)w;
        total *= (size_t)n;
        if (n == 1) continue;
        if (k && st[k - 1] == n * s) {
            sh[k - 1] *= n;
            st[k - 1] = s;
        } else {
            sh[k] = n;
            st[k++] = s;
        }
    }
    if (!k) {
        sh[0] = 1;
        st[0] = (long long)w;
        k = 1;
    }
    PyObject *out = PyErr_Occurred() ? NULL : PyBytes_FromStringAndSize(NULL, (Py_ssize_t)total);
    if (out && total) {
        unsigned char *dst = (unsigned char *)PyBytes_AsString(out); /* a new object: still ours to fill */
        int contiguous = st[k - 1] == (long long)w, tiled = !contiguous && k >= 2 && llabs(st[k - 2]) < llabs(st[k - 1]);
        Py_ssize_t outer = k - 1 - tiled; /* the axes walked here; each step copies the rest */
        size_t chunk = (size_t)sh[k - 1] * (tiled ? (size_t)sh[k - 2] : 1) * w;
        long long off = 0;
        Py_BEGIN_ALLOW_THREADS
        for (Py_ssize_t a = 0; a < outer; a++) idx[a] = 0;
        for (size_t done = 0; done < total; done += chunk, dst += chunk) {
            if (contiguous) memcpy(dst, base + off, chunk);
            else if (tiled) gather_tiles(dst, base + off, st[k - 2], st[k - 1], (size_t)sh[k - 2], (size_t)sh[k - 1], w);
            else gather_elements(dst, base + off, st[k - 1], (size_t)sh[k - 1], w);
            for (Py_ssize_t a = outer - 1; a >= 0; a--) { /* the next outer index, the last axis fastest */
                off += st[a];
                if (++idx[a] < sh[a]) break;
                off -= st[a] * sh[a];
                idx[a] = 0;
            }
        }
        Py_END_ALLOW_THREADS
    }
    PyMem_Free(sh);
    return out;
}

/* DLPack's structures, as DLPack 1.3's dlpack.h lays them out, for reading a producer's capsule
   (_dlpack.py): dlpack_open takes the tensor, gather copies it and dlpack_close hands it back. */
typedef struct {
    int32_t device_type, device_id;
} dl_device;

typedef struct {
    uint8_t code, bits;
    uint16_t lanes;
} dl_dtype;

typedef struct {
    void *data;
    dl_device device;
    int32_t ndim;
    dl_dtype dtype;
    int64_t *shape, *strides; /* strides in elements, NULL for row-major */
    uint64_t byte_offset;
} dl_tensor;

typedef struct dl_managed {
    dl_tensor tensor;
    void *manager_ctx;
    void (*deleter)(struct dl_managed *self);
} dl_managed;

typedef struct dl_managed_versioned {
    struct {
        uint32_t major, minor;
    } version;
    void *manager_ctx;
    void (*deleter)(struct dl_managed_versioned *self);
    uint64_t flags;
    dl_tensor tensor;
} dl_managed_versioned;

/* A taken tensor, held by an owner capsule until dlpack_close (or the owner's end) calls its deleter. */
typedef struct {
    void *managed; /* NULL once handed back */
    int versioned;
} dl_owner;

static void dl_release(dl_owner *o) {
    if (o->versioned) {
        dl_managed_versioned *x = o->managed;
        if (x && x->deleter) x->deleter(x);
    } else {
        dl_managed *x = o->managed;
        if (x && x->deleter) x->deleter(x);
    }
    o->managed = NULL;
}

static void dl_owner_end(PyObject *owner) {
    dl_owner *o = PyCapsule_GetPointer(owner, "exactref.dlpack");
    if (!o) return;
    dl_release(o);
    PyMem_Free(o);
}

/* An unused DLPack capsule's tensor, taken (the capsule renamed "used_...", so it no longer frees
   it): (owner, (major version or None for a legacy capsule, dtype code, bits, lanes, shape,
   strides or None, the data's address)); for a major version other than 1, whose layout differs,
   (owner, (major version, None, ...)). */
static PyObject *py_dlpack_open(PyObject *self, PyObject *args) {
    PyObject *capsule;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    int versioned = PyCapsule_IsValid(capsule, "dltensor_versioned");
    if (!versioned && !PyCapsule_IsValid(capsule, "dltensor")) {
        PyErr_SetString(PyExc_ValueError, "__dlpack__ did not return an unused DLPack capsule");
        return NULL;
    }
    void *managed = PyCapsule_GetPointer(capsule, versioned ? "dltensor_versioned" : "dltensor");
    if (!managed) return NULL;
    dl_owner *o = PyMem_Malloc(sizeof *o);
    if (!o) return PyErr_NoMemory();
    o->managed = NULL;
    o->versioned = versioned;
    PyObject *owner = PyCapsule_New(o, "exactref.dlpack", dl_owner_end);
    if (!owner) {
        PyMem_Free(o);
        return NULL;
    }
    if (PyCapsule_SetName(capsule, versioned ? "used_dltensor_versioned" : "used_dltensor")) {
        Py_DECREF(owner);
        PyErr_SetString(PyExc_RuntimeError, "could not take ownership of the DLPack capsule");
        return NULL;
    }
    o->managed = managed;
    if (versioned && ((dl_managed_versioned *)managed)->version.major != 1) {
        /* another major version's layout: no field but the version and the deleter may be read */
        PyObject *out = Py_BuildValue("(O(NOOOOOO))", owner, PyLong_FromUnsignedLong(((dl_managed_versioned *)managed)->version.major), Py_None, Py_None,
                                      Py_None, Py_None, Py_None, Py_None);
        Py_DECREF(owner); /* where out failed, this hands the tensor back */
        return out;
    }
    const dl_tensor *t = versioned ? &((dl_managed_versioned *)managed)->tensor : &((dl_managed *)managed)->tensor;
    PyObject *major = versioned ? PyLong_FromUnsignedLong(((dl_managed_versioned *)managed)->version.major) : Py_None;
    if (!versioned) Py_INCREF(major);
    PyObject *dims[2] = {PyTuple_New(t->ndim), t->strides ? PyTuple_New(t->ndim) : Py_None}; /* shape, strides */
    if (!t->strides) Py_INCREF(Py_None);
    int ok = major && dims[0] && dims[1];
    for (int32_t a = 0; ok && a < t->ndim; a++) {
        PyObject *n = PyLong_FromLongLong(t->shape[a]);
        ok = n && !PyTuple_SetItem(dims[0], a, n); /* SetItem takes n, also where it fails */
        if (ok && t->strides) {
            PyObject *s = PyLong_FromLongLong(t->strides[a]);
            ok = s && !PyTuple_SetItem(dims[1], a, s);
        }
    }
    void *data = t->data ? (void *)((unsigned char *)t->data + t->byte_offset) : NULL;
    PyObject *out = ok ? Py_BuildValue("(O(OiiiOON))", owner, major, (int)t->dtype.code, (int)t->dtype.bits, (int)t->dtype.lanes, dims[0], dims[1],
                                       PyLong_FromVoidPtr(data))
                       : NULL;
    Py_XDECREF(major);
    Py_XDECREF(dims[0]);
    Py_XDECREF(dims[1]);
    Py_DECREF(owner); /* where out failed, this hands the tensor back */
    return out;
}

static PyObject *py_dlpack_close(PyObject *self, PyObject *args) {
    PyObject *owner;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &owner)) return NULL;
    dl_owner *o = PyCapsule_GetPointer(owner, "exactref.dlpack");
    if (!o) return NULL;
    dl_release(o);
    return Py_NewRef(Py_None);
}

/* The calling thread's floating-point environment (csrc/system.c): fenv_save() as bytes, or None
   where it cannot be saved; fenv_reset() whether it is now the default; fenv_restore(saved). */
static PyObject *py_fenv_save(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)fenv_size());
    if (!out) return NULL;
    if (fenv_save((unsigned char *)PyBytes_AsString(out))) { /* a new object: still ours to fill */
        Py_DECREF(out);
        return Py_NewRef(Py_None);
    }
    return out;
}

static PyObject *py_fenv_reset(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyBool_FromLong(!fenv_reset());
}

static PyObject *py_fenv_restore(PyObject *self, PyObject *args) {
    PyObject *saved;
    (void)self;
    if (!PyArg_ParseTuple(args, "S", &saved)) return NULL;
    return PyBool_FromLong(!fenv_restore((const unsigned char *)PyBytes_AsString(saved)));
}

/* A coefficient: (m, e) for a real one, or ((m, e), (m, e)) its real and imaginary parts. */
static int coefficient_load(PyObject *obj, dyad v[2], int *parts) {
    if (!PyTuple_Check(obj) || PyTuple_Size(obj) != 2) {
        PyErr_SetString(PyExc_TypeError, "a coefficient is (m, e) or ((m, e), (m, e))");
        return -1;
    }
    *parts = PyTuple_Check(PyTuple_GetItem(obj, 0)) ? 2 : 1;
    if (*parts == 1) return dyad_load_clamped(obj, &v[0]);
    return dyad_load_clamped(PyTuple_GetItem(obj, 0), &v[0]) || dyad_load_clamped(PyTuple_GetItem(obj, 1), &v[1]) ? -1 : 0;
}

/* gemm's epilogue: (alpha, beta, c, big_endian, c_layout, c_terms, c_radius, c_marks, f64_layout,
   part), c_terms how a unit beta's c enters the sum (TERMS_*; gemm_c_terms). */
static int epilogue_load(epilogue *x, PyObject *obj) {
    gemm_fold_init(&x->fold);
    dyad_init(&x->alpha);
    dyad_init(&x->beta);
    for (int k = 0; k < 4; k++) big_init(&x->nan[k]);
    x->c = x->c_radius = x->c_marks = NULL;
    x->on = obj != Py_None;
    if (!x->on) return 0;
    PyObject *alpha, *beta, *c, *layout, *radius, *marks, *f64;
    Py_ssize_t len;
    int unit, parts_a, parts_b, rc;
    dyad va[2], vb[2];
    for (int k = 0; k < 2; k++) {
        dyad_init(&va[k]);
        dyad_init(&vb[k]);
    }
    x->on = 0; /* until the channels are made */
    rc = !PyArg_ParseTuple(obj, "OOOpOiOOOi", &alpha, &beta, &c, &x->c_big, &layout, &unit, &radius, &marks, &f64, &x->part) ||
         coefficient_load(alpha, va, &parts_a) || coefficient_load(beta, vb, &parts_b);
    if (!rc && parts_a != parts_b) {
        PyErr_SetString(PyExc_ValueError, "alpha and beta must both be real or both complex");
        rc = -1;
    }
    if (!rc && (rc = dyad_copy(&x->alpha, &va[0]) || dyad_copy(&x->beta, &vb[0]) || gemm_channels_set(&x->ch, va, vb, parts_a))) error(EXACTREF_NOMEM);
    for (int k = 0; k < 2; k++) {
        dyad_free(&va[k]);
        dyad_free(&vb[k]);
    }
    if (rc) return -1;
    x->on = 1;
    x->c_terms = gemm_c_terms(&x->ch, unit);
    if (partfmt_load(layout, &x->cf, x->nan) || partfmt_load(f64, &x->f64, x->nan + 2) || (c != Py_None && data_load(c, &x->c, &len)) ||
        (radius != Py_None && data_load(radius, &x->c_radius, &len)) || (marks != Py_None && data_load(marks, &x->c_marks, &len)))
        return -1;
    return 0;
}

/* Complex gemm takes B's parts' operands as a pair, everything else one operand. */
static int epilogue_parts_check(const epilogue *x, const operand *const b[2]) {
    if ((x->on && x->ch.parts == 2) == (b[1] != NULL)) return 0;
    PyErr_SetString(PyExc_TypeError, "complex gemm takes B's parts' operands as a pair, everything else one operand");
    return -1;
}

/* B's operands: a capsule, or for complex gemm the pair of its parts' operands. */
static int b_parts_load(const operand *b[2], PyObject *obj) {
    b[1] = NULL;
    if (PyTuple_Check(obj)) {
        if (PyTuple_Size(obj) != 2) {
            PyErr_SetString(PyExc_TypeError, "B's operands are a capsule or a pair of capsules");
            return -1;
        }
        b[0] = operand_get(PyTuple_GetItem(obj, 0));
        b[1] = b[0] ? operand_get(PyTuple_GetItem(obj, 1)) : NULL;
        return b[1] ? 0 : -1;
    }
    b[0] = operand_get(obj);
    return b[0] ? 0 : -1;
}

/* An operand's extent for Lemma T's slack (mma_extent): (may be subnormal, largest (m, e)). */
static PyObject *py_mma_extent(PyObject *self, PyObject *args) {
    PyObject *data_obj, *layout, *f64_layout, *radii_obj;
    int big, radii_big;
    (void)self;
    if (!PyArg_ParseTuple(args, "SpOOOp", &data_obj, &big, &layout, &f64_layout, &radii_obj, &radii_big)) return NULL;
    partfmt f, f64;
    bigint nan[4];
    dyad largest;
    const unsigned char *data, *radii = NULL;
    Py_ssize_t len, rlen;
    int sub = 0;
    PyObject *out = NULL;
    for (int k = 0; k < 4; k++) big_init(&nan[k]);
    dyad_init(&largest);
    if (!partfmt_load(layout, &f, nan) && !partfmt_load(f64_layout, &f64, nan + 2) && !data_load(data_obj, &data, &len) &&
        (radii_obj == Py_None || !data_load(radii_obj, &radii, &rlen))) {
        int rc = mma_extent(&sub, &largest, &f, data, big, (size_t)len / part_size(&f), &f64, radii, radii_big);
        PyObject *m = rc ? NULL : dyad_store(&largest, 0);
        if (rc)
            error(rc);
        else if (m)
            out = Py_BuildValue("(NN)", PyBool_FromLong(sub), m);
    }
    for (int k = 0; k < 4; k++) big_free(&nan[k]);
    dyad_free(&largest);
    return out;
}

/* Lemma T's sigma from two extents (mma_slack): (m, e). */
static PyObject *py_mma_slack(PyObject *self, PyObject *args) {
    long long emin;
    int sub_x, sub_y;
    PyObject *x_obj, *y_obj;
    (void)self;
    if (!PyArg_ParseTuple(args, "LpOpO", &emin, &sub_x, &x_obj, &sub_y, &y_obj)) return NULL;
    dyad x, y, sigma;
    dyad_init(&x);
    dyad_init(&y);
    dyad_init(&sigma);
    PyObject *out = NULL;
    if (!dyad_load_clamped(x_obj, &x) && !dyad_load_clamped(y_obj, &y)) {
        int rc = mma_slack(&sigma, emin, sub_x, &x, sub_y, &y);
        out = rc ? error(rc) : dyad_store(&sigma, 0);
    }
    dyad_free(&x);
    dyad_free(&y);
    dyad_free(&sigma);
    return out;
}

static PyObject *py_mma_terms(PyObject *self, PyObject *args) {
    long long N, f, e_f, split, K, extra, steps, w;
    PyObject *reduction, *sigma_obj, *alpha_obj = Py_None;
    (void)self;
    if (!PyArg_ParseTuple(args, "LLLLOOLLLL|O", &N, &f, &e_f, &split, &reduction, &sigma_obj, &K, &extra, &steps, &w, &alpha_obj)) return NULL;
    floatfmt r;
    bigint nan[2];
    dyad sigma, G, H, alpha;
    big_init(&nan[0]);
    big_init(&nan[1]);
    dyad_init(&sigma);
    dyad_init(&G);
    dyad_init(&H);
    dyad_init(&alpha);
    PyObject *out = NULL;
    if ((reduction == Py_None || !layout_load(reduction, &r, nan)) && !dyad_load_clamped(sigma_obj, &sigma) &&
        (alpha_obj == Py_None || !dyad_load_clamped(alpha_obj, &alpha))) {
        int rc = mma_terms(&G, &H, N, f, e_f, split, reduction == Py_None ? NULL : &r, &sigma, K, extra, steps, w);
        if (!rc && alpha_obj != Py_None) rc = mma_alpha(&H, &alpha, w);
        if (rc) {
            error(rc);
        } else {
            PyObject *g = dyad_store(&G, 0), *h = g ? dyad_store(&H, 0) : NULL;
            out = h ? Py_BuildValue("(OO)", g, h) : NULL;
            Py_XDECREF(g);
            Py_XDECREF(h);
        }
    }
    big_free(&nan[0]);
    big_free(&nan[1]);
    dyad_free(&sigma);
    dyad_free(&G);
    dyad_free(&H);
    dyad_free(&alpha);
    return out;
}

static PyObject *py_order_depth(PyObject *self, PyObject *args) {
    int kind;
    Py_ssize_t k, b;
    (void)self;
    if (!PyArg_ParseTuple(args, "inn", &kind, &k, &b)) return NULL;
    return PyLong_FromLongLong(product_order_depth(kind, (size_t)k, (size_t)b));
}

static PyObject *py_scan_flagged(PyObject *self, PyObject *args) {
    long long top, low, bits;
    int any, nonfinite;
    (void)self;
    if (!PyArg_ParseTuple(args, "LLppL", &top, &low, &any, &nonfinite, &bits)) return NULL;
    return PyBool_FromLong(scan_flagged(top, low, any, nonfinite, bits));
}

static PyObject *py_product_reference(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj, *a_bases, *b_bases, *layout, *epi_obj;
    Py_ssize_t m_, n_, o0, o1;
    int rnd, ovf, elementwise;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOnnnnOiipO", &a_obj, &b_obj, &a_bases, &b_bases, &m_, &n_, &o0, &o1, &layout, &rnd, &ovf, &elementwise, &epi_obj))
        return NULL;
    const operand *a = operand_get(a_obj), *bs[2] = {NULL, NULL};
    partfmt pf = {0};
    bigint nan[2];
    addressing where;
    epilogue epi;
    big_init(&nan[0]);
    big_init(&nan[1]);
    int loaded = !epilogue_load(&epi, epi_obj);
    if (!loaded || !a || b_parts_load(bs, b_obj) || epilogue_parts_check(&epi, bs) || partfmt_load(layout, &pf, nan) ||
        addressing_load(&where, a_bases, b_bases, m_, n_)) {
        epilogue_free(&epi);
        big_free(&nan[0]);
        big_free(&nan[1]);
        return NULL;
    }
    size_t count = (size_t)(o1 - o0), itemsize = part_size(&pf);
    PyObject *encodings = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(count * itemsize));
    PyObject *flags = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)count);
    PyObject *mids = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(count * ENTRY_WIDTH));
    PyObject *out = NULL;
    if (encodings && flags && mids) {
        unsigned char *enc = (unsigned char *)PyBytes_AsString(encodings), *fl = (unsigned char *)PyBytes_AsString(flags);
        unsigned char *mid = (unsigned char *)PyBytes_AsString(mids);
        int rc;
        size_t bad = 0;
        bigint bad_value;
        big_init(&bad_value);
        Py_BEGIN_ALLOW_THREADS
        rc = product_reference_run(enc, fl, mid, a, bs, &where, &epi, &pf, rnd, ovf, elementwise, (size_t)o0, count, &bad, &bad_value);
        Py_END_ALLOW_THREADS
        if (rc == EXACTREF_INT_OVERFLOW)
            out = Py_BuildValue("(OnN)", Py_None, (Py_ssize_t)bad, int_store(&bad_value));
        else if (rc)
            error(rc);
        else
            out = Py_BuildValue("(OOO)", encodings, flags, mids);
        big_free(&bad_value);
    }
    Py_XDECREF(encodings);
    Py_XDECREF(flags);
    Py_XDECREF(mids);
    epilogue_free(&epi);
    big_free(&nan[0]);
    big_free(&nan[1]);
    return out;
}

static int mma_args_load(mma_args *x, PyObject *obj) {
    dyad *all[] = {&x->G, &x->H, &x->top, &x->thresholds[0], &x->thresholds[1], &x->G2, &x->H2};
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) dyad_init(all[i]);
    big_init(&x->nan[0]);
    big_init(&x->nan[1]);
    x->on = obj != Py_None;
    x->count = 1;
    x->epilogue = 0;
    if (!x->on) return 0;
    PyObject *G, *H, *reduction, *G2 = Py_None, *H2 = Py_None;
    if (!PyArg_ParseTuple(obj, "OOO|OO", &G, &H, &reduction, &G2, &H2) || dyad_load_clamped(G, &x->G) || dyad_load_clamped(H, &x->H)) return -1;
    if (G2 != Py_None) {
        if (dyad_load_clamped(G2, &x->G2) || dyad_load_clamped(H2, &x->H2)) return -1;
        x->epilogue = 1;
    }
    int rc = floatfmt_set(&x->f32, 8, 23, SPECIAL_IEEE, NULL, 1);
    if (!rc) rc = floatfmt_threshold(&x->thresholds[0], &x->top, &x->f32);
    x->has_infinity = 1;
    if (!rc && reduction != Py_None) {
        if (layout_load(reduction, &x->reduction, x->nan)) return -1;
        rc = floatfmt_threshold(&x->thresholds[1], NULL, &x->reduction);
        x->count = 2;
        x->has_infinity = x->reduction.special == SPECIAL_IEEE;
    }
    if (rc) {
        error(rc);
        return -1;
    }
    return 0;
}

static PyObject *py_product_int_bounds(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj, *a_bases, *b_bases, *layout, *f64_layout, *refs_obj, *radii_obj, *epi_obj;
    Py_ssize_t m_, n_, o0, o1;
    int ovf;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOnnnnOOiSOO", &a_obj, &b_obj, &a_bases, &b_bases, &m_, &n_, &o0, &o1, &layout, &f64_layout, &ovf, &refs_obj,
                          &radii_obj, &epi_obj))
        return NULL;
    const operand *a = operand_get(a_obj), *b = operand_get(b_obj), *ra = NULL, *rb = NULL;
    if (radii_obj != Py_None) {
        PyObject *x, *y;
        if (!PyArg_ParseTuple(radii_obj, "OO", &x, &y)) return NULL;
        if (x != Py_None && !(ra = operand_get(x))) return NULL;
        if (y != Py_None && !(rb = operand_get(y))) return NULL;
    }
    partfmt pf;
    floatfmt f64;
    bigint nan[4];
    addressing where;
    epilogue epi;
    const unsigned char *refs;
    Py_ssize_t len;
    for (int k = 0; k < 4; k++) big_init(&nan[k]);
    int loaded = !epilogue_load(&epi, epi_obj);
    if (!loaded || !a || !b || partfmt_load(layout, &pf, nan) || layout_load(f64_layout, &f64, nan + 2) || data_load(refs_obj, &refs, &len) ||
        addressing_load(&where, a_bases, b_bases, m_, n_)) {
        epilogue_free(&epi);
        for (int k = 0; k < 4; k++) big_free(&nan[k]);
        return NULL;
    }
    size_t count = (size_t)(o1 - o0), itemsize = part_size(&pf);
    PyObject *tols = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(8 * count));
    PyObject *lows = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(count * itemsize));
    PyObject *highs = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(count * itemsize));
    PyObject *flags = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)count), *out = NULL;
    if (tols && lows && highs && flags) {
        unsigned char *t = (unsigned char *)PyBytes_AsString(tols), *lo_out = (unsigned char *)PyBytes_AsString(lows);
        unsigned char *hi_out = (unsigned char *)PyBytes_AsString(highs), *fl = (unsigned char *)PyBytes_AsString(flags);
        int rc;
        Py_BEGIN_ALLOW_THREADS
        rc = product_int_bounds_run(t, lo_out, hi_out, fl, a, b, ra, rb, &where, &epi, &pf, &f64, ovf, refs, (size_t)o0, count);
        Py_END_ALLOW_THREADS
        if (rc)
            error(rc);
        else
            out = Py_BuildValue("(OOOO)", tols, lows, highs, flags);
    }
    Py_XDECREF(tols);
    Py_XDECREF(lows);
    Py_XDECREF(highs);
    Py_XDECREF(flags);
    epilogue_free(&epi);
    for (int k = 0; k < 4; k++) big_free(&nan[k]);
    return out;
}

static PyObject *py_product_normal(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj, *a_bases, *b_bases, *compute_layout, *layout, *epi_obj;
    Py_ssize_t m_, n_, o0, o1;
    int flush, rnd, ovf, exact_terms;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOnnnnOpOiiOi", &a_obj, &b_obj, &a_bases, &b_bases, &m_, &n_, &o0, &o1, &compute_layout, &flush, &layout, &rnd,
                          &ovf, &epi_obj, &exact_terms))
        return NULL;
    const operand *a = operand_get(a_obj), *bs[2] = {NULL, NULL};
    floatfmt compute, f;
    bigint nan[4];
    addressing where;
    epilogue epi;
    for (int k = 0; k < 4; k++) big_init(&nan[k]);
    int loaded = !epilogue_load(&epi, epi_obj);
    if (!loaded || !a || b_parts_load(bs, b_obj) || epilogue_parts_check(&epi, bs) || layout_load(compute_layout, &compute, nan) ||
        layout_load(layout, &f, nan + 2) || addressing_load(&where, a_bases, b_bases, m_, n_)) {
        epilogue_free(&epi);
        for (int k = 0; k < 4; k++) big_free(&nan[k]);
        return NULL;
    }
    size_t count = (size_t)(o1 - o0), itemsize = (size_t)(1 + f.exp_bits + f.man_bits) / 8;
    PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(count * itemsize));
    if (out) {
        unsigned char *enc = (unsigned char *)PyBytes_AsString(out);
        int rc;
        Py_BEGIN_ALLOW_THREADS
        rc = product_normal_run(enc, a, bs, &where, &epi, &compute, flush, exact_terms, &f, rnd, ovf, (size_t)o0, count);
        Py_END_ALLOW_THREADS
        if (rc) {
            Py_CLEAR(out);
            error(rc);
        }
    }
    epilogue_free(&epi);
    for (int k = 0; k < 4; k++) big_free(&nan[k]);
    return out;
}

/* A named algorithm's outputs (MatmulAlgorithm.run, product_algorithm_run) */
static PyObject *py_product_run(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj, *a_bases, *b_bases, *acc_layout, *layout;
    Py_ssize_t m_, n_, o0, o1, block;
    int acc_rnd, method, fma, rnd, ovf;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOnnnnOiinpOii", &a_obj, &b_obj, &a_bases, &b_bases, &m_, &n_, &o0, &o1, &acc_layout, &acc_rnd, &method, &block,
                          &fma, &layout, &rnd, &ovf))
        return NULL;
    const operand *a = operand_get(a_obj), *b = operand_get(b_obj);
    floatfmt acc, f;
    bigint nan[4];
    addressing where;
    for (int k = 0; k < 4; k++) big_init(&nan[k]);
    if (!a || !b || layout_load(acc_layout, &acc, nan) || layout_load(layout, &f, nan + 2) || addressing_load(&where, a_bases, b_bases, m_, n_)) {
        for (int k = 0; k < 4; k++) big_free(&nan[k]);
        return NULL;
    }
    size_t count = (size_t)(o1 - o0), itemsize = (size_t)(1 + f.exp_bits + f.man_bits) / 8;
    PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(count * itemsize));
    if (out) {
        unsigned char *enc = (unsigned char *)PyBytes_AsString(out);
        int rc;
        Py_BEGIN_ALLOW_THREADS
        rc = product_algorithm_run(enc, a, b, &where, &acc, acc_rnd, method, block > 0 ? (size_t)block : 1, fma, &f, rnd, ovf, (size_t)o0, count);
        Py_END_ALLOW_THREADS
        if (rc) {
            Py_CLEAR(out);
            error(rc);
        }
    }
    for (int k = 0; k < 4; k++) big_free(&nan[k]);
    return out;
}

static PyObject *py_product_bounds(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj, *a_bases, *b_bases, *compute_layout, *layout, *f64_layout, *refs_obj, *row_obj, *col_obj, *epi_obj, *order_obj, *mma_obj;
    PyObject *radii_obj, *reach_obj;
    Py_ssize_t m_, n_, o0, o1;
    int flush, normal, exact_terms, rnd, ovf, few, order_kind = 0;
    long long w, block = 0;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOnnnnOppiOOiiSOOLOOOpOO", &a_obj, &b_obj, &a_bases, &b_bases, &m_, &n_, &o0, &o1, &compute_layout, &flush,
                          &normal, &exact_terms, &layout, &f64_layout, &rnd, &ovf, &refs_obj, &row_obj, &col_obj, &w, &epi_obj, &order_obj, &mma_obj,
                          &few, &radii_obj, &reach_obj))
        return NULL;
    /* a coefficient product's reach (product_reach): A's marks per slot, b"" for none; None for the plain rule */
    const unsigned char *reach_marks = NULL;
    Py_ssize_t reach_len = 0;
    int reach = reach_obj != Py_None;
    if (reach && data_load(reach_obj, &reach_marks, &reach_len)) return NULL;
    if (!reach_len) reach_marks = NULL;
    /* the operands' radii, read as the operands are: (ra, rb), each a capsule or None (rb a pair as B's operands are), or None */
    const operand *ra = NULL, *rbs[2] = {NULL, NULL};
    if (radii_obj != Py_None) {
        PyObject *x, *y;
        if (!PyArg_ParseTuple(radii_obj, "OO", &x, &y)) return NULL;
        if (x != Py_None && !(ra = operand_get(x))) return NULL;
        if (y != Py_None && b_parts_load(rbs, y)) return NULL;
    }
    /* a named order: (kind, block), kind 1 sequential, 2 pairwise, 3 blocked (block its block size), 4 exact accumulation, 5 the
       16-slot algorithm (block its accumulator's rounding) */
    if (order_obj != Py_None && !PyArg_ParseTuple(order_obj, "iL", &order_kind, &block)) return NULL;
    const operand *a = operand_get(a_obj), *bs[2] = {NULL, NULL};
    floatfmt compute, f, f64;
    bigint nan[6];
    addressing where;
    epilogue epi;
    mma_args mma;
    for (int k = 0; k < 6; k++) big_init(&nan[k]);
    const unsigned char *refs, *row_marks = NULL, *col_marks = NULL;
    Py_ssize_t len;
    PyObject *out = NULL;
    int loaded = !epilogue_load(&epi, epi_obj);
    loaded = !mma_args_load(&mma, mma_obj) && loaded;
    if (!loaded || !a || b_parts_load(bs, b_obj) || epilogue_parts_check(&epi, bs) || layout_load(compute_layout, &compute, nan) ||
        layout_load(layout, &f, nan + 2) || layout_load(f64_layout, &f64, nan + 4) || data_load(refs_obj, &refs, &len) || (row_obj != Py_None && data_load(row_obj, &row_marks, &len)) ||
        (col_obj != Py_None && data_load(col_obj, &col_marks, &len)) || addressing_load(&where, a_bases, b_bases, m_, n_)) {
        epilogue_free(&epi);
        mma_args_free(&mma);
        for (int k = 0; k < 6; k++) big_free(&nan[k]);
        return NULL;
    }
    size_t count = (size_t)(o1 - o0), itemsize = (size_t)(1 + f.exp_bits + f.man_bits) / 8;
    PyObject *tols = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(8 * count));
    PyObject *lows = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(count * itemsize));
    PyObject *highs = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(count * itemsize));
    PyObject *flags = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)count);
    if (tols && lows && highs && flags) {
        unsigned char *t = (unsigned char *)PyBytes_AsString(tols), *lo_out = (unsigned char *)PyBytes_AsString(lows);
        unsigned char *hi_out = (unsigned char *)PyBytes_AsString(highs), *fl = (unsigned char *)PyBytes_AsString(flags);
        int rc;
        Py_BEGIN_ALLOW_THREADS
        rc = product_bounds_run(t, lo_out, hi_out, fl, a, bs, ra, rbs, &where, &epi, &mma, &compute, &f, &f64, flush, normal, exact_terms, rnd, ovf, few,
                                order_kind, block, w, refs, row_marks, col_marks, reach, reach_marks, (size_t)o0, count);
        Py_END_ALLOW_THREADS
        if (rc)
            error(rc);
        else
            out = Py_BuildValue("(OOOO)", tols, lows, highs, flags);
    }
    Py_XDECREF(tols);
    Py_XDECREF(lows);
    Py_XDECREF(highs);
    Py_XDECREF(flags);
    epilogue_free(&epi);
    mma_args_free(&mma);
    for (int k = 0; k < 6; k++) big_free(&nan[k]);
    return out;
}

/* The attention forward's reference (attention.c) */
#define ATT_TABLES_CAPSULE "exactref._c.att_tables"

/* The capsule holds two table sets: the scores' and a float mask's (empty without one). */
static void att_tables_destroy(PyObject *capsule) {
    att_tables *t = PyCapsule_GetPointer(capsule, ATT_TABLES_CAPSULE);
    if (t) {
        att_tables_free(&t[0]);
        att_tables_free(&t[1]);
        free(t);
    }
}

/* A float mask's finest unit 2**v (the least lowest set bit of its finite nonzero values) and the
   digits its tables need: distances below 133 (beyond, a weight is below 2**-191) and below the
   span of its finite values, in units of 2**v. *digits > 8 where they would not fit the kernel's
   words. */
static int mask_digits(int64_t *v, int *digits, const att_call *a, size_t count) {
    int any = 0, rc = 0;
    dyad lo, hi, x;
    dyad_init(&lo);
    dyad_init(&hi);
    dyad_init(&x);
    *v = 0;
    *digits = 0;
    for (size_t at = 0; !rc && at < count; at++) {
        int kind, neg;
        uint64_t sig;
        int64_t e;
        att_mask_read(&kind, &neg, &sig, &e, a, at);
        if (kind != VALUE_FINITE) continue;
        if ((rc = big_set_word(&x.m, sig, neg && sig))) break;
        x.e = e;
        if (sig) {
            int64_t low = e;
            for (uint64_t t = sig; !(t & 1); t >>= 1) low++;
            if (!any || low < *v) *v = low;
        }
        if (!any || dyad_cmp(&x, &lo) < 0) rc = dyad_copy(&lo, &x);
        if (!rc && (!any || dyad_cmp(&x, &hi) > 0)) rc = dyad_copy(&hi, &x);
        any = 1;
    }
    if (!rc && any) {
        /* the span, capped at 133: bits of span / 2**v */
        dyad cap;
        dyad_init(&cap);
        if (lo.m.n) lo.m.neg = !lo.m.neg;
        if (!(rc = dyad_add_exact(&x, &hi, &lo)) && !(rc = dyad_set_si(&cap, 133, 0)) && dyad_cmp(&x, &cap) > 0) rc = dyad_copy(&x, &cap);
        if (!rc && x.m.n) {
            int64_t bits = x.e + big_bits(&x.m) - *v; /* none at or below 0: every distance is below one unit, so 0 */
            *digits = bits > 0 ? (int)((bits + ATT_B - 1) / ATT_B) : 0;
        }
        dyad_free(&cap);
    }
    dyad_free(&lo);
    dyad_free(&hi);
    dyad_free(&x);
    return rc;
}

/* sigma as None (1/sqrt(E)) or (num, den) */
static int sigma_load(PyObject *obj, bigint *num, bigint *den) {
    PyObject *a, *b;
    if (obj == Py_None) return 0;
    if (!PyArg_ParseTuple(obj, "OO", &a, &b)) return -1;
    return int_load(a, num) || int_load(b, den) ? -1 : 0;
}

/* A float mask's format and byte order, (layout, big), into the call (None: no float mask). */
static int mask_load(PyObject *obj, att_call *a) {
    PyObject *layout;
    int big;
    if (obj == Py_None) return 0;
    if (!PyArg_ParseTuple(obj, "Op", &layout, &big)) return -1;
    partfmt mf = {0};
    bigint mnan[2];
    big_init(&mnan[0]);
    big_init(&mnan[1]);
    int bad = partfmt_load(layout, &mf, mnan);
    big_free(&mnan[0]);
    big_free(&mnan[1]);
    if (bad) return -1;
    a->mask_float = 1;
    a->mask_fmt = mf.f;
    a->mask_fmt.nan[0] = a->mask_fmt.nan[1] = NULL; /* only decoded: its NaN encodings not kept */
    a->mask_size = part_size(&mf);
    a->mask_big = big;
    return 0;
}

/* Over an operand's rows with a nonzero value, the least lowest set bit and the largest top. */
static int operand_span(const operand *x, int64_t *low, int64_t *top) {
    int any = 0;
    for (size_t r = 0; r < x->rows; r++) {
        if (!x->max_sig[r]) continue;
        if (!any || x->low[r] < *low) *low = x->low[r];
        if (!any || x->top[r] > *top) *top = x->top[r];
        any = 1;
    }
    return any;
}

static PyObject *py_attention_tables(PyObject *self, PyObject *args) {
    PyObject *qo, *ko, *vo, *sig, *mask_obj, *mask_layout;
    Py_ssize_t E;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOnOO", &qo, &ko, &vo, &sig, &E, &mask_obj, &mask_layout)) return NULL;
    int64_t ql = 0, qt = 0, kl = 0, kt = 0, u = 0, bits = 0;
    int spans = 0;
    if (PyTuple_Check(qo)) {
        /* the operands read in parts (operand_scan's statistics): (Q's low, Q's top, K's low, K's top, both have a nonzero
           value, a slot of Q, K or V flagged or two words) */
        long long a0, a1, a2, a3;
        int blocked;
        if (!PyArg_ParseTuple(qo, "LLLLpp", &a0, &a1, &a2, &a3, &spans, &blocked)) return NULL;
        if (blocked) return Py_NewRef(Py_None);
        ql = a0, qt = a1, kl = a2, kt = a3;
    } else {
        const operand *q = operand_get(qo), *k = operand_get(ko), *v = vo == Py_None ? NULL : operand_get(vo); /* v None: the backward's */
        if (!q || !k || (vo != Py_None && !v)) return NULL;
        if (q->flagged_count || k->flagged_count || q->slots_hi || k->slots_hi || (v && (v->flagged_count || v->slots_hi))) return Py_NewRef(Py_None);
        spans = operand_span(q, &ql, &qt) && operand_span(k, &kl, &kt);
    }
    bigint num, den;
    big_init(&num);
    big_init(&den);
    if (sigma_load(sig, &num, &den)) {
        big_free(&num);
        big_free(&den);
        return NULL;
    }
    /* the call's finest score unit 2**u, and the bits of its widest distance:
       |score| < E 2**(top_q + top_k), so a distance is below 2 E 2**(top_q + top_k - u) */
    if (spans) {
        u = ql + kl;
        bits = word_bits((uint64_t)E) + 1 + qt + kt - u;
    }
    /* a float mask's unit and digits */
    int64_t mv = 0;
    int mdigits = 0, mask_rc = 0;
    if (mask_layout != Py_None) {
        att_call m;
        memset(&m, 0, sizeof m);
        const unsigned char *mdata;
        Py_ssize_t mlen;
        if (mask_load(mask_layout, &m) || data_load(mask_obj, &mdata, &mlen)) {
            mask_rc = -1;
        } else {
            m.mask = mdata;
            mask_rc = mask_digits(&mv, &mdigits, &m, (size_t)mlen / m.mask_size);
            if (mask_rc) error(mask_rc);
        }
        if (mask_rc) {
            big_free(&num);
            big_free(&den);
            return NULL;
        }
    }
    PyObject *out = NULL;
    if (bits > 127 || bits > (int64_t)ATT_B * 8 || mdigits > 8) {
        Py_INCREF(Py_None);
        out = Py_None;
    } else {
        int digits = den.n && !num.n ? 0 : (int)((bits + ATT_B - 1) / ATT_B);
        att_tables *t = malloc(2 * sizeof *t);
        int rc = EXACTREF_NOMEM;
        if (t) {
            bigint one;
            big_init(&one);
            att_tables_init(&t[1]);
            Py_BEGIN_ALLOW_THREADS
            rc = att_tables_build(&t[0], &num, &den, (int64_t)E, u, digits);
            /* the mask's tables: exp(-2**v d 2**(B i)), sigma 1 */
            if (!rc && !(rc = big_set_word(&one, 1, 0))) rc = att_tables_build(&t[1], &one, &one, (int64_t)E, mv, mdigits);
            Py_END_ALLOW_THREADS
            big_free(&one);
        }
        if (rc) {
            if (t) {
                att_tables_free(&t[0]);
                att_tables_free(&t[1]);
            }
            free(t);
            error(rc);
        } else {
            PyObject *capsule = PyCapsule_New(t, ATT_TABLES_CAPSULE, att_tables_destroy);
            if (!capsule) {
                att_tables_free(&t[0]);
                att_tables_free(&t[1]);
                free(t);
            } else {
                out = Py_BuildValue("(NLi)", capsule, (long long)u, digits);
            }
        }
    }
    big_free(&num);
    big_free(&den);
    return out;
}

static PyObject *py_attention_reference(PyObject *self, PyObject *args) {
    PyObject *qo, *ko, *vo, *to, *sig, *rows_obj, *mask_obj, *layout, *lse_layout;
    Py_ssize_t E, u, kr0, vr0, S, Ev, p, cap;
    int causal, rnd, ovf, lrnd, lovf;
    (void)self;
    PyObject *mask_layout;
    if (!PyArg_ParseTuple(args, "OOOOOnnOnnnnpOOOiiOiinn", &qo, &ko, &vo, &to, &sig, &E, &u, &rows_obj, &kr0, &vr0, &S, &Ev, &causal, &mask_obj,
                          &mask_layout, &layout, &rnd, &ovf, &lse_layout, &lrnd, &lovf, &p, &cap))
        return NULL;
    att_call a;
    memset(&a, 0, sizeof a);
    a.q = operand_get(qo);
    a.k = operand_get(ko);
    a.v = operand_get(vo);
    a.tables = to == Py_None ? NULL : PyCapsule_GetPointer(to, ATT_TABLES_CAPSULE); /* None: any operand, Lemma X per weight */
    if (!a.q || !a.k || !a.v || (to != Py_None && !a.tables)) return NULL;
    partfmt pf = {0}, lf = {0};
    bigint nan[2], lnan[2];
    for (int i = 0; i < 2; i++) {
        big_init(&nan[i]);
        big_init(&lnan[i]);
    }
    big_init(&a.num);
    big_init(&a.den);
    const unsigned char *rows;
    Py_ssize_t len;
    PyObject *out = NULL, *res[8] = {NULL};
    const unsigned char *mask_data = NULL;
    Py_ssize_t mask_len;
    if (sigma_load(sig, &a.num, &a.den) || partfmt_load(layout, &pf, nan) || partfmt_load(lse_layout, &lf, lnan) || data_load(rows_obj, &rows, &len) ||
        (mask_obj != Py_None && data_load(mask_obj, &mask_data, &mask_len)) || mask_load(mask_layout, &a))
        goto done;
    if (mask_obj != Py_None) a.mask = mask_data;
    a.mtables = a.tables ? a.tables + 1 : NULL;
    a.S = (size_t)S;
    a.Ev = (size_t)Ev;
    a.causal = causal;
    a.sign = !a.den.n ? 1 : !a.num.n ? 0 : a.num.neg ? -1 : 1;
    a.E = E;
    a.u = u;
    a.p = p;
    a.cap = cap;
    a.out = pf.f;
    a.lse = lf.f;
    a.out_size = part_size(&pf);
    a.lse_size = part_size(&lf);
    a.rnd = rnd;
    a.ovf = ovf;
    a.lrnd = lrnd;
    a.lovf = lovf;
    size_t count = (size_t)len / 32, cells = count * a.Ev;
    Py_ssize_t sizes[8] = {(Py_ssize_t)(cells * a.out_size), (Py_ssize_t)cells, (Py_ssize_t)(cells * ENTRY_WIDTH), (Py_ssize_t)cells,
                           (Py_ssize_t)(count * a.lse_size), (Py_ssize_t)count, (Py_ssize_t)(count * ENTRY_WIDTH), (Py_ssize_t)count};
    for (int i = 0; i < 8; i++)
        if (!(res[i] = PyBytes_FromStringAndSize(NULL, sizes[i]))) goto done;
    unsigned char *o[8];
    for (int i = 0; i < 8; i++) o[i] = (unsigned char *)PyBytes_AsString(res[i]);
    memset(o[0], 0, (size_t)sizes[0]);
    memset(o[4], 0, (size_t)sizes[4]);
    int rc = 0;
    Py_BEGIN_ALLOW_THREADS
    att_row w;
    att_row_init(&w);
    entry *x = malloc((a.Ev ? a.Ev : 1) * sizeof *x), lx;
    if (!x) rc = EXACTREF_NOMEM;
    for (size_t r = 0; !rc && r < count; r++) {
        int64_t qr, i, moff, mstep;
        memcpy(&qr, rows + 32 * r, 8);
        memcpy(&i, rows + 32 * r + 8, 8);
        memcpy(&moff, rows + 32 * r + 16, 8);
        memcpy(&mstep, rows + 32 * r + 24, 8);
        memset(x, 0, a.Ev * sizeof *x);
        memset(&lx, 0, sizeof lx);
        rc = att_reference_row(&w, &a, (size_t)qr, (size_t)i, (size_t)moff, (size_t)mstep, (size_t)kr0, (size_t)vr0, o[0] + r * a.Ev * a.out_size, x, o[3] + r * a.Ev,
                               o[4] + r * a.lse_size, &lx, o[7] + r);
        for (size_t c = 0; !rc && c < a.Ev; c++) {
            o[1][r * a.Ev + c] = x[c].flag;
            memcpy(o[2] + (r * a.Ev + c) * ENTRY_WIDTH, x[c].mid, ENTRY_WIDTH);
        }
        o[5][r] = lx.flag;
        memcpy(o[6] + r * ENTRY_WIDTH, lx.mid, ENTRY_WIDTH);
    }
    free(x);
    att_row_free(&w);
    Py_END_ALLOW_THREADS
    if (rc)
        error(rc);
    else
        out = Py_BuildValue("(OOOOOOOO)", res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]);
done:
    for (int i = 0; i < 8; i++) Py_XDECREF(res[i]);
    for (int i = 0; i < 2; i++) {
        big_free(&nan[i]);
        big_free(&lnan[i]);
    }
    big_free(&a.num);
    big_free(&a.den);
    return out;
}

/* The attention forward's tolerance (attention_tol.c) */
#define ATT_TOL_CAPSULE "exactref._c.att_tol"

static void att_tol_destroy(PyObject *capsule) {
    att_tol *t = PyCapsule_GetPointer(capsule, ATT_TOL_CAPSULE);
    if (t) {
        att_tol_free(t);
        free(t);
    }
}

static PyObject *py_attention_tolerance(PyObject *self, PyObject *args) {
    PyObject *qo, *ko, *vo, *vto, *to, *tolo, *sig, *rows_obj, *mask_obj, *layout, *lse_layout, *refs, *lse_refs, *eflags, *emids, *lflags, *lmids;
    PyObject *rqo, *rko, *vvo, *vro, *vunbo, *mrado;
    Py_ssize_t E, u, kr0, vr0, S, Ev;
    int causal, rnd, ovf, lrnd, lovf;
    long long ve;
    (void)self;
    PyObject *mask_layout;
    if (!PyArg_ParseTuple(args, "OOOOOOOnnOnnnnpOOOiiOiiOOOOOOOOOOLOO", &qo, &ko, &vo, &vto, &to, &tolo, &sig, &E, &u, &rows_obj, &kr0, &vr0, &S, &Ev,
                          &causal, &mask_obj, &mask_layout, &layout, &rnd, &ovf, &lse_layout, &lrnd, &lovf, &refs, &lse_refs, &eflags, &emids, &lflags,
                          &lmids, &rqo, &rko, &vvo, &vro, &ve, &vunbo, &mrado))
        return NULL;
    att_call a;
    memset(&a, 0, sizeof a);
    a.mask = NULL;
    a.mask_float = 0;
    /* inputs that are earlier results: their radii */
    if ((rqo != Py_None && !(a.rq = operand_get(rqo))) || (rko != Py_None && !(a.rk = operand_get(rko)))) return NULL;
    Py_ssize_t radii_len;
    if ((vvo != Py_None && (data_load(vvo, &a.vv, &radii_len) || data_load(vro, &a.vr, &radii_len) || data_load(vunbo, &a.vunb, &radii_len))) ||
        (mrado != Py_None && data_load(mrado, &a.mrad, &radii_len)))
        return NULL;
    a.ve = ve;
    a.uncertain = rqo != Py_None || rko != Py_None || vvo != Py_None || mrado != Py_None;
    a.q = operand_get(qo);
    a.k = operand_get(ko);
    a.v = operand_get(vo);
    a.vt = operand_get(vto);
    a.tables = to == Py_None ? NULL : PyCapsule_GetPointer(to, ATT_TABLES_CAPSULE); /* None: Lemma X per weight */
    const att_tol *t = PyCapsule_GetPointer(tolo, ATT_TOL_CAPSULE);
    if (!a.q || !a.k || !a.v || !a.vt || (to != Py_None && !a.tables) || !t) return NULL;
    partfmt pf = {0}, lf = {0};
    bigint nan[2], lnan[2];
    for (int z = 0; z < 2; z++) {
        big_init(&nan[z]);
        big_init(&lnan[z]);
    }
    big_init(&a.num);
    big_init(&a.den);
    const unsigned char *rows, *rf, *lrf, *ef, *em, *lfl, *lm;
    Py_ssize_t len, n1, n2, n3, n4, n5, n6;
    PyObject *out = NULL, *res[9] = {NULL};
    const unsigned char *mask_data = NULL;
    Py_ssize_t mask_len;
    if (sigma_load(sig, &a.num, &a.den) || partfmt_load(layout, &pf, nan) || partfmt_load(lse_layout, &lf, lnan) || data_load(rows_obj, &rows, &len) ||
        data_load(refs, &rf, &n1) || data_load(lse_refs, &lrf, &n2) || data_load(eflags, &ef, &n3) || data_load(emids, &em, &n4) ||
        data_load(lflags, &lfl, &n5) || data_load(lmids, &lm, &n6) || (mask_obj != Py_None && data_load(mask_obj, &mask_data, &mask_len)))
        goto done;
    if (mask_obj != Py_None) a.mask = mask_data;
    if (mask_load(mask_layout, &a)) goto done;
    a.mtables = a.tables ? a.tables + 1 : NULL;
    a.S = (size_t)S;
    a.Ev = (size_t)Ev;
    a.causal = causal;
    a.sign = !a.den.n ? 1 : !a.num.n ? 0 : a.num.neg ? -1 : 1;
    a.E = E;
    a.u = u;
    a.p = t->w;
    a.out = pf.f;
    a.lse = lf.f;
    a.out_size = part_size(&pf);
    a.lse_size = part_size(&lf);
    a.rnd = rnd;
    a.ovf = ovf;
    a.lrnd = lrnd;
    a.lovf = lovf;
    size_t count = (size_t)len / 32, cells = count * a.Ev;
    Py_ssize_t sizes[9] = {(Py_ssize_t)count, (Py_ssize_t)(cells * 8), (Py_ssize_t)(cells * a.out_size), (Py_ssize_t)(cells * a.out_size),
                           (Py_ssize_t)cells, (Py_ssize_t)(count * 8), (Py_ssize_t)(count * a.lse_size), (Py_ssize_t)(count * a.lse_size),
                           (Py_ssize_t)count};
    for (int z = 0; z < 9; z++)
        if (!(res[z] = PyBytes_FromStringAndSize(NULL, sizes[z]))) goto done;
    unsigned char *o[9];
    for (int z = 0; z < 9; z++) o[z] = (unsigned char *)PyBytes_AsString(res[z]);
    for (int z = 1; z < 9; z++) memset(o[z], 0, (size_t)sizes[z]);
    int rc = 0;
    Py_BEGIN_ALLOW_THREADS
    att_row w;
    att_row_init(&w);
    entry *x = malloc((a.Ev ? a.Ev : 1) * sizeof *x), lx;
    if (!x) rc = EXACTREF_NOMEM;
    for (size_t r = 0; !rc && r < count; r++) {
        int64_t qr, i, moff, mstep;
        memcpy(&qr, rows + 32 * r, 8);
        memcpy(&i, rows + 32 * r + 8, 8);
        memcpy(&moff, rows + 32 * r + 16, 8);
        memcpy(&mstep, rows + 32 * r + 24, 8);
        for (size_t c = 0; c < a.Ev; c++) {
            x[c].flag = ef[r * a.Ev + c];
            memcpy(x[c].mid, em + (r * a.Ev + c) * ENTRY_WIDTH, ENTRY_WIDTH);
        }
        lx.flag = lfl[r];
        memcpy(lx.mid, lm + r * ENTRY_WIDTH, ENTRY_WIDTH);
        rc = att_tolerance_row(&w, &a, t, (size_t)qr, (size_t)i, (size_t)moff, (size_t)mstep, (size_t)kr0, (size_t)vr0, x, rf + r * a.Ev * a.out_size, &lx,
                               lrf + r * a.lse_size,
                               o[1] + r * a.Ev * 8, o[2] + r * a.Ev * a.out_size, o[3] + r * a.Ev * a.out_size, o[4] + r * a.Ev, o[5] + r * 8,
                               o[6] + r * a.lse_size, o[7] + r * a.lse_size, o[8] + r, o[0] + r);
    }
    free(x);
    att_row_free(&w);
    Py_END_ALLOW_THREADS
    (void)n1;
    (void)n2;
    (void)n3;
    (void)n4;
    (void)n5;
    (void)n6;
    if (rc)
        error(rc);
    else
        out = Py_BuildValue("(OOOOOOOOO)", res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8]);
done:
    for (int z = 0; z < 9; z++) Py_XDECREF(res[z]);
    for (int z = 0; z < 2; z++) {
        big_free(&nan[z]);
        big_free(&lnan[z]);
    }
    big_free(&a.num);
    big_free(&a.den);
    return out;
}

/* The step-by-step program (attention_tol.c) */
#define ATT_PROG_CAPSULE "exactref._c.att_prog"

static void att_prog_destroy(PyObject *capsule) {
    att_prog *g = PyCapsule_GetPointer(capsule, ATT_PROG_CAPSULE);
    if (g) {
        att_prog_free(g);
        free(g);
    }
}

/* The forward's settings (attention_tol.c's att_settings) */
#define ATT_SETTINGS_CAPSULE "exactref._c.att_settings"

static void att_settings_destroy(PyObject *capsule) {
    att_settings *s = PyCapsule_GetPointer(capsule, ATT_SETTINGS_CAPSULE);
    if (s) {
        att_settings_free(s);
        free(s);
    }
}

/* a float format from its layout, without its NaN encodings (the constants read none) */
static int fmt_load(PyObject *layout, floatfmt *f) {
    bigint nan[2];
    big_init(&nan[0]);
    big_init(&nan[1]);
    int bad = layout_load(layout, f, nan);
    f->nan[0] = f->nan[1] = NULL;
    big_free(&nan[0]);
    big_free(&nan[1]);
    return bad;
}

#define MEXP_CAPSULE "exactref.mexp_batch"

static void mexp_destroy(PyObject *capsule) {
    mexp_batch *b = PyCapsule_GetPointer(capsule, MEXP_CAPSULE);
    if (b) {
        mexp_batch_free(b);
        free(b);
    }
}

/* matrix_exp_new(data, layout, big_endian, count, n, parts, L, box): the exponentials of `count`
   matrices n x n (row-major, a complex element's parts in turn) of an array's bytes, at L limbs;
   box (m, e) or None widens every entry part by that radius. A matrix with a non-finite entry has no
   result. */
static PyObject *py_matrix_exp_new(PyObject *self, PyObject *args) {
    PyObject *data, *layout, *box;
    int big_endian, parts, L;
    Py_ssize_t count, n;
    (void)self;
    if (!PyArg_ParseTuple(args, "SOpnniiO", &data, &layout, &big_endian, &count, &n, &parts, &L, &box)) return NULL;
    if (L < 1 || L > MAT_LIMBS_MAX || (parts != 1 && parts != 2) || count < 0 || n < 0) {
        PyErr_SetString(PyExc_ValueError, "matrix_exp_new: limbs, parts or sizes");
        return NULL;
    }
    partfmt f;
    bigint nan[2];
    big_init(&nan[0]);
    big_init(&nan[1]);
    const unsigned char *src;
    Py_ssize_t len;
    mexp_batch *b = calloc(1, sizeof *b);
    value v;
    dyad r;
    dyad_init(&v.v);
    dyad_init(&r);
    PyObject *out = NULL;
    int rc = 0, has_box = box != Py_None;
    if (!b) {
        PyErr_NoMemory();
        goto done;
    }
    if (partfmt_load(layout, &f, nan) || data_load(data, &src, &len) || (has_box && dyad_load(box, &r))) goto done;
    if ((size_t)len < (size_t)count * (size_t)n * (size_t)n * (size_t)parts * part_size(&f)) {
        PyErr_SetString(PyExc_ValueError, "matrix_exp_new: the data");
        goto done;
    }
    Py_BEGIN_ALLOW_THREADS
    rc = mexp_batch_init(b, (size_t)count, (size_t)n, L, parts);
    for (size_t m = 0; !rc && m < (size_t)count; m++) {
        int finite = 1;
        for (size_t at = 0; !rc && at < (size_t)n * (size_t)n; at++)
            for (int p = 0; !rc && p < parts; p++) {
                if ((rc = part_read(&v, &f, src, (m * (size_t)n * (size_t)n + at) * (size_t)parts + (size_t)p, big_endian))) break;
                if (v.kind != VALUE_FINITE) {
                    finite = 0;
                    continue;
                }
                rc = imat_set(&b->in[m], p, at / (size_t)n, at % (size_t)n, &v.v, has_box && v.v.m.n ? &r : NULL); /* zeros stay exact */
            }
        if (!finite) { /* no result: an empty matrix the batch skips */
            imat_free(&b->in[m]);
            if (!rc) rc = imat_init(&b->in[m], 0, 0, L, parts);
        }
    }
    Py_END_ALLOW_THREADS
    if (rc) {
        error(rc);
        goto done;
    }
    if ((out = PyCapsule_New(b, MEXP_CAPSULE, mexp_destroy))) b = NULL;
done:
    if (b) {
        mexp_batch_free(b);
        free(b);
    }
    big_free(&nan[0]);
    big_free(&nan[1]);
    dyad_free(&v.v);
    dyad_free(&r);
    return out;
}

/* matrix_exp_rational(data, layout, big_endian, n, parts, wanted): for one matrix n x n of an array's bytes (finite
   entries), and the wanted (i, j, part) triples, each one's exact value (num, den) where it is rational, else None
   (matrix_exp.md, Rational parts) */
static PyObject *py_matrix_exp_rational(PyObject *self, PyObject *args) {
    PyObject *data, *layout, *wanted;
    int big_endian, parts;
    Py_ssize_t n;
    (void)self;
    if (!PyArg_ParseTuple(args, "SOpniO", &data, &layout, &big_endian, &n, &parts, &wanted)) return NULL;
    PyObject *seq = PySequence_Tuple(wanted);
    if (!seq) return NULL;
    Py_ssize_t count = PyTuple_Size(seq);
    partfmt f;
    bigint nan[2];
    big_init(&nan[0]);
    big_init(&nan[1]);
    const unsigned char *src;
    Py_ssize_t len;
    size_t nn = (size_t)n * (size_t)n, total = (size_t)parts * nn;
    dyad *A = calloc(total + 1, sizeof *A);
    size_t *w = malloc((3 * (size_t)count + 1) * sizeof *w);
    int *rational = calloc((size_t)count + 1, sizeof *rational);
    rat *values = calloc((size_t)count + 1, sizeof *values); /* zeroed: freed safely before their init */
    value v;
    dyad_init(&v.v);
    PyObject *out = NULL;
    int rc = 0;
    if (!A || !w || !rational || !values) {
        PyErr_NoMemory();
        goto done;
    }
    for (size_t a = 0; a < total; a++) dyad_init(&A[a]);
    for (Py_ssize_t a = 0; a < count; a++) rat_init(&values[a]);
    if (partfmt_load(layout, &f, nan) || data_load(data, &src, &len)) goto done;
    for (Py_ssize_t a = 0; a < count; a++) {
        Py_ssize_t i, j;
        int part;
        if (!PyArg_ParseTuple(PyTuple_GetItem(seq, a), "nni", &i, &j, &part)) goto done;
        if (i < 0 || j < 0 || i >= n || j >= n || part < 0 || part >= parts) {
            PyErr_SetString(PyExc_IndexError, "matrix_exp_rational");
            goto done;
        }
        w[3 * a] = (size_t)i;
        w[3 * a + 1] = (size_t)j;
        w[3 * a + 2] = (size_t)part;
    }
    for (size_t at = 0; !rc && at < nn; at++)
        for (int p = 0; !rc && p < parts; p++) {
            if ((rc = part_read(&v, &f, src, at * (size_t)parts + (size_t)p, big_endian))) break;
            if (v.kind != VALUE_FINITE) {
                PyErr_SetString(PyExc_ValueError, "matrix_exp_rational: a non-finite entry");
                goto done;
            }
            rc = dyad_copy(&A[(size_t)p * nn + at], &v.v);
        }
    if (!rc) {
        Py_BEGIN_ALLOW_THREADS
        rc = mexp_rational(rational, values, (size_t)n, parts, A, w, (size_t)count);
        Py_END_ALLOW_THREADS
    }
    if (rc) {
        error(rc);
        goto done;
    }
    out = PyList_New(count);
    for (Py_ssize_t a = 0; out && a < count; a++) {
        PyObject *item;
        if (!rational[a]) {
            item = Py_NewRef(Py_None);
        } else {
            PyObject *num = int_store(&values[a].num), *den = num ? int_store(&values[a].den) : NULL;
            item = den ? PyTuple_Pack(2, num, den) : NULL;
            Py_XDECREF(num);
            Py_XDECREF(den);
        }
        if (!item) {
            Py_CLEAR(out);
            break;
        }
        PyList_SetItem(out, a, item);
    }
done:
    Py_DECREF(seq);
    if (A)
        for (size_t a = 0; a < total; a++) dyad_free(&A[a]);
    if (values)
        for (Py_ssize_t a = 0; a < count; a++) rat_free(&values[a]);
    free(A);
    free(w);
    free(rational);
    free(values);
    dyad_free(&v.v);
    big_free(&nan[0]);
    big_free(&nan[1]);
    return out;
}

#define MREF_CAPSULE "exactref.mref"

static void mref_destroy(PyObject *capsule) {
    mref *x = PyCapsule_GetPointer(capsule, MREF_CAPSULE);
    if (x) {
        mref_free(x);
        free(x);
    }
}

/* matrix_exp_reference(a, layout, big_endian, count, n, parts, e, e_big_endian, out_layout, rounding, overflow, start,
   cap): matrix_exp's reference for `count` matrices n x n of an array's bytes (row-major, a complex element's parts in
   turn); with e (bytes of the same layout and shape, or None), the top-right block of exp([[A, E], [0, A]]) */
static PyObject *py_matrix_exp_reference(PyObject *self, PyObject *args) {
    PyObject *a_data, *layout, *e_data, *out_layout;
    int big_endian, e_big, parts, rnd, ovf;
    Py_ssize_t count, n;
    long long start, cap;
    (void)self;
    if (!PyArg_ParseTuple(args, "SOpnniOpOiiLL", &a_data, &layout, &big_endian, &count, &n, &parts, &e_data, &e_big, &out_layout, &rnd, &ovf, &start,
                          &cap))
        return NULL;
    partfmt f, out;
    bigint nan[2], onan[2];
    big_init(&nan[0]);
    big_init(&nan[1]);
    big_init(&onan[0]);
    big_init(&onan[1]);
    const unsigned char *src, *esrc = NULL;
    Py_ssize_t len, elen = 0;
    mref *x = calloc(1, sizeof *x);
    value v;
    dyad_init(&v.v);
    PyObject *out_obj = NULL;
    int rc = 0, block = e_data != Py_None;
    size_t N = (size_t)n, M = block ? 2 * N : N, nn = N * N;
    if (!x) {
        PyErr_NoMemory();
        goto done;
    }
    if (partfmt_load(layout, &f, nan) || partfmt_load(out_layout, &out, onan) || data_load(a_data, &src, &len) || (block && data_load(e_data, &esrc, &elen)))
        goto done;
    if ((parts != 1 && parts != 2) || (size_t)len < (size_t)count * nn * (size_t)parts * part_size(&f) ||
        (block && (size_t)elen < (size_t)count * nn * (size_t)parts * part_size(&f))) {
        PyErr_SetString(PyExc_ValueError, "matrix_exp_reference: the data");
        goto done;
    }
    Py_BEGIN_ALLOW_THREADS
    rc = mref_init(x, (size_t)count, M, parts, 0, N, block ? N : 0, M, &out.f, part_size(&out), rnd, ovf, start, cap);
    for (size_t m = 0; !rc && m < (size_t)count; m++) {
        x->finite[m] = 1;
        for (size_t i = 0; !rc && i < N; i++)
            for (size_t j = 0; !rc && j < N; j++)
                for (int p = 0; !rc && p < parts; p++) {
                    if ((rc = part_read(&v, &f, src, (m * nn + i * N + j) * (size_t)parts + (size_t)p, big_endian))) break;
                    if (v.kind != VALUE_FINITE) {
                        x->finite[m] = 0;
                        continue;
                    }
                    dyad *plane = &x->A[(m * (size_t)parts + (size_t)p) * M * M];
                    if ((rc = dyad_copy(&plane[i * M + j], &v.v))) break;
                    if (block && (rc = dyad_copy(&plane[(i + N) * M + j + N], &v.v))) break;
                    if (!block) continue;
                    if ((rc = part_read(&v, &f, esrc, (m * nn + i * N + j) * (size_t)parts + (size_t)p, e_big))) break;
                    if (v.kind != VALUE_FINITE) {
                        x->finite[m] = 0;
                        continue;
                    }
                    rc = dyad_copy(&plane[i * M + j + N], &v.v);
                }
        if (!rc && block && x->finite[m]) rc = mref_balance(x, m);
    }
    Py_END_ALLOW_THREADS
    if (rc) {
        error(rc);
        goto done;
    }
    if ((out_obj = PyCapsule_New(x, MREF_CAPSULE, mref_destroy))) x = NULL;
done:
    if (x) {
        mref_free(x);
        free(x);
    }
    dyad_free(&v.v);
    big_free(&nan[0]);
    big_free(&nan[1]);
    big_free(&onan[0]);
    big_free(&onan[1]);
    return out_obj;
}

/* matrix_exp_reference_next(reference): the tiles of the next phase, 0 when done */
static PyObject *py_matrix_exp_reference_next(PyObject *self, PyObject *args) {
    PyObject *capsule;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    mref *x = PyCapsule_GetPointer(capsule, MREF_CAPSULE);
    if (!x) return NULL;
    size_t tiles;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = mref_next(x, &tiles);
    Py_END_ALLOW_THREADS
    return rc ? error(rc) : PyLong_FromSize_t(tiles);
}

/* matrix_exp_reference_tile(reference, t): runs tile t of the current phase */
static PyObject *py_matrix_exp_reference_tile(PyObject *self, PyObject *args) {
    PyObject *capsule;
    Py_ssize_t t;
    (void)self;
    if (!PyArg_ParseTuple(args, "On", &capsule, &t)) return NULL;
    mref *x = PyCapsule_GetPointer(capsule, MREF_CAPSULE);
    if (!x) return NULL;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = mref_tile(x, (size_t)t);
    Py_END_ALLOW_THREADS
    if (rc) return error(rc);
    return Py_NewRef(Py_None);
}

/* matrix_exp_reference_result(reference): (encodings, entry flags, entry midpoints, statuses) as bytes, per wanted part
   in row-major order */
static PyObject *py_matrix_exp_reference_result(PyObject *self, PyObject *args) {
    PyObject *capsule;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    mref *x = PyCapsule_GetPointer(capsule, MREF_CAPSULE);
    if (!x) return NULL;
    size_t slots = x->count * (x->r1 - x->r0) * (x->c1 - x->c0) * (size_t)x->parts;
    PyObject *enc = PyBytes_FromStringAndSize((const char *)x->enc, (Py_ssize_t)(slots * x->size));
    PyObject *flags = enc ? PyBytes_FromStringAndSize(NULL, (Py_ssize_t)slots) : NULL;
    PyObject *mids = flags ? PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(slots * ENTRY_WIDTH)) : NULL;
    PyObject *status = mids ? PyBytes_FromStringAndSize((const char *)x->status, (Py_ssize_t)slots) : NULL;
    if (!status) {
        Py_XDECREF(enc);
        Py_XDECREF(flags);
        Py_XDECREF(mids);
        return NULL;
    }
    unsigned char *fl = (unsigned char *)PyBytes_AsString(flags), *md = (unsigned char *)PyBytes_AsString(mids); /* new objects: ours to fill */
    for (size_t s = 0; s < slots; s++) {
        fl[s] = x->entries[s].flag;
        memcpy(md + s * ENTRY_WIDTH, x->entries[s].mid, ENTRY_WIDTH);
    }
    return Py_BuildValue("(NNNN)", enc, flags, mids, status);
}

#define MTOL_CAPSULE "exactref.mtol"

static void mtol_destroy(PyObject *capsule) {
    mtol *x = PyCapsule_GetPointer(capsule, MTOL_CAPSULE);
    if (x) {
        mtol_free(x);
        free(x);
    }
}

/* matrix_exp_tolerance(a, layout, big_endian, count, n, parts, e, e_big_endian, ra, ra_big, re, re_big, f64_layout,
   compute_layout, flush, allowance (num, den), out_layout, rounding, overflow, refs): matrix_exp's tolerance for the
   stored values a (and with e the top-right block of exp([[A, E], [0, A]])), each matrix's entries widened by the
   radii ra (and re, for E) when given (float64 parts of f64_layout); refs the reference's encodings of the wanted
   parts. */
static PyObject *py_matrix_exp_tolerance(PyObject *self, PyObject *args) {
    PyObject *a_data, *layout, *e_data, *ra_data, *re_data, *f64_layout, *compute_layout, *allow, *out_layout, *refs;
    int big_endian, e_big, ra_big, re_big, flush, parts, rnd, ovf;
    Py_ssize_t count, n;
    (void)self;
    if (!PyArg_ParseTuple(args, "SOpnniOpOpOpOOpOOiiS", &a_data, &layout, &big_endian, &count, &n, &parts, &e_data, &e_big, &ra_data, &ra_big, &re_data,
                          &re_big, &f64_layout, &compute_layout, &flush, &allow, &out_layout, &rnd, &ovf, &refs))
        return NULL;
    partfmt f, f64, F, out;
    bigint nan[8], anum, aden;
    for (int k = 0; k < 8; k++) big_init(&nan[k]);
    big_init(&anum);
    big_init(&aden);
    const unsigned char *src, *esrc = NULL, *rasrc = NULL, *resrc = NULL, *rsrc;
    Py_ssize_t len, elen = 0, ralen = 0, relen = 0, rlen;
    mtol *x = calloc(1, sizeof *x);
    value v;
    dyad_init(&v.v);
    PyObject *out_obj = NULL;
    int rc = 0, block = e_data != Py_None, radii = ra_data != Py_None || re_data != Py_None;
    size_t N = (size_t)n, M = block ? 2 * N : N, nn = N * N;
    if (!x) {
        PyErr_NoMemory();
        goto done;
    }
    if (partfmt_load(layout, &f, nan) || partfmt_load(f64_layout, &f64, nan + 2) || partfmt_load(compute_layout, &F, nan + 4) ||
        partfmt_load(out_layout, &out, nan + 6) || data_load(a_data, &src, &len) || (block && data_load(e_data, &esrc, &elen)) ||
        (ra_data != Py_None && data_load(ra_data, &rasrc, &ralen)) || (re_data != Py_None && data_load(re_data, &resrc, &relen)) ||
        data_load(refs, &rsrc, &rlen))
        goto done;
    if (!PyTuple_Check(allow) || PyTuple_Size(allow) != 2) {
        PyErr_SetString(PyExc_TypeError, "matrix_exp_tolerance: the allowance is a pair (num, den)");
        goto done;
    }
    if (int_load(PyTuple_GetItem(allow, 0), &anum) || int_load(PyTuple_GetItem(allow, 1), &aden)) goto done;
    size_t wanted = (size_t)count * nn * (size_t)parts, psize = part_size(&f);
    if ((parts != 1 && parts != 2) || (size_t)len < (size_t)count * nn * (size_t)parts * psize || (block && (size_t)elen < (size_t)count * nn * (size_t)parts * psize) ||
        (rasrc && (size_t)ralen < (size_t)count * nn * (size_t)parts * 8) || (resrc && (size_t)relen < (size_t)count * nn * (size_t)parts * 8) ||
        (size_t)rlen < wanted * part_size(&out)) {
        PyErr_SetString(PyExc_ValueError, "matrix_exp_tolerance: the data");
        goto done;
    }
    Py_BEGIN_ALLOW_THREADS
    rc = mtol_init(x, (size_t)count, M, parts, radii, 0, N, block ? N : 0, M, &F.f, flush, &anum, &aden, &out.f, part_size(&out), rnd, ovf, &f64.f, rsrc);
    for (size_t m = 0; !rc && m < (size_t)count; m++) {
        x->finite[m] = x->bounded[m] = 1;
        for (size_t i = 0; !rc && i < N; i++)
            for (size_t j = 0; !rc && j < N; j++)
                for (int p = 0; !rc && p < parts; p++) {
                    size_t at = (m * nn + i * N + j) * (size_t)parts + (size_t)p;
                    dyad *plane = &x->A[(m * (size_t)parts + (size_t)p) * M * M], *rplane = radii ? &x->R[(m * (size_t)parts + (size_t)p) * M * M] : NULL;
                    /* a value and its radius into the block's places: (i, j) and (i + N, j + N) for A, (i, j + N) for E */
                    for (int which = 0; !rc && which < (block ? 2 : 1); which++) {
                        const unsigned char *vs = which ? esrc : src, *rs = which ? resrc : rasrc;
                        int vbig = which ? e_big : big_endian, rbig = which ? re_big : ra_big;
                        size_t at1 = which ? i * M + j + N : i * M + j;
                        /* the radius first: an unbounded one leaves no bound even where the stored value is not finite */
                        if (rs) {
                            if ((rc = part_read(&v, &f64, rs, at, rbig))) break;
                            if (v.kind != VALUE_FINITE) {
                                x->bounded[m] = 0;
                            } else {
                                if ((rc = dyad_copy(&rplane[at1], &v.v))) break;
                                if (block && !which && (rc = dyad_copy(&rplane[(i + N) * M + j + N], &v.v))) break;
                            }
                        }
                        if ((rc = part_read(&v, &f, vs, at, vbig))) break;
                        if (v.kind != VALUE_FINITE) {
                            x->finite[m] = 0;
                            continue;
                        }
                        if ((rc = dyad_copy(&plane[at1], &v.v))) break;
                        if (block && !which) rc = dyad_copy(&plane[(i + N) * M + j + N], &v.v);
                    }
                }
    }
    Py_END_ALLOW_THREADS
    if (rc) {
        error(rc);
        goto done;
    }
    if ((out_obj = PyCapsule_New(x, MTOL_CAPSULE, mtol_destroy))) x = NULL;
done:
    if (x) {
        mtol_free(x);
        free(x);
    }
    dyad_free(&v.v);
    for (int k = 0; k < 8; k++) big_free(&nan[k]);
    big_free(&anum);
    big_free(&aden);
    return out_obj;
}

/* matrix_exp_tolerance_next(tolerance): the tiles of the next phase, 0 when done */
static PyObject *py_matrix_exp_tolerance_next(PyObject *self, PyObject *args) {
    PyObject *capsule;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    mtol *x = PyCapsule_GetPointer(capsule, MTOL_CAPSULE);
    if (!x) return NULL;
    size_t tiles;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = mtol_next(x, &tiles);
    Py_END_ALLOW_THREADS
    return rc ? error(rc) : PyLong_FromSize_t(tiles);
}

/* matrix_exp_tolerance_tile(tolerance, t): runs tile t of the current phase */
static PyObject *py_matrix_exp_tolerance_tile(PyObject *self, PyObject *args) {
    PyObject *capsule;
    Py_ssize_t t;
    (void)self;
    if (!PyArg_ParseTuple(args, "On", &capsule, &t)) return NULL;
    mtol *x = PyCapsule_GetPointer(capsule, MTOL_CAPSULE);
    if (!x) return NULL;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = mtol_tile(x, (size_t)t);
    Py_END_ALLOW_THREADS
    if (rc) return error(rc);
    return Py_NewRef(Py_None);
}

/* matrix_exp_tolerance_result(tolerance): (tolerances, lows, highs, flags) as bytes, per wanted part row-major */
static PyObject *py_matrix_exp_tolerance_result(PyObject *self, PyObject *args) {
    PyObject *capsule;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    mtol *x = PyCapsule_GetPointer(capsule, MTOL_CAPSULE);
    if (!x) return NULL;
    size_t slots = x->count * (x->r1 - x->r0) * (x->c1 - x->c0) * (size_t)x->parts;
    PyObject *tol = PyBytes_FromStringAndSize((const char *)x->tol, (Py_ssize_t)(slots * 8));
    PyObject *low = tol ? PyBytes_FromStringAndSize((const char *)x->low, (Py_ssize_t)(slots * x->size)) : NULL;
    PyObject *high = low ? PyBytes_FromStringAndSize((const char *)x->high, (Py_ssize_t)(slots * x->size)) : NULL;
    PyObject *flags = high ? PyBytes_FromStringAndSize((const char *)x->flags, (Py_ssize_t)slots) : NULL;
    if (!flags) {
        Py_XDECREF(tol);
        Py_XDECREF(low);
        Py_XDECREF(high);
        return NULL;
    }
    return Py_BuildValue("(NNNN)", tol, low, high, flags);
}

/* matrix_exp_next(batch): the tiles of the next phase, 0 when every exponential is done */
static PyObject *py_matrix_exp_next(PyObject *self, PyObject *args) {
    PyObject *capsule;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    mexp_batch *b = PyCapsule_GetPointer(capsule, MEXP_CAPSULE);
    if (!b) return NULL;
    size_t tiles;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = mexp_batch_next(b, &tiles);
    Py_END_ALLOW_THREADS
    return rc ? error(rc) : PyLong_FromSize_t(tiles);
}

/* matrix_exp_tile(batch, t): runs tile t of the current phase */
static PyObject *py_matrix_exp_tile(PyObject *self, PyObject *args) {
    PyObject *capsule;
    Py_ssize_t t;
    (void)self;
    if (!PyArg_ParseTuple(args, "On", &capsule, &t)) return NULL;
    mexp_batch *b = PyCapsule_GetPointer(capsule, MEXP_CAPSULE);
    if (!b) return NULL;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = mexp_batch_tile(b, (size_t)t);
    Py_END_ALLOW_THREADS
    if (rc) return error(rc);
    return Py_NewRef(Py_None);
}

/* matrix_exp_ends(batch, m, part, i, j): ((m, e), (m, e)), the ends of matrix m's entry, or None
   where it has no result */
static PyObject *py_matrix_exp_ends(PyObject *self, PyObject *args) {
    PyObject *capsule;
    Py_ssize_t m, i, j;
    int part;
    (void)self;
    if (!PyArg_ParseTuple(args, "Oninn", &capsule, &m, &part, &i, &j)) return NULL;
    mexp_batch *b = PyCapsule_GetPointer(capsule, MEXP_CAPSULE);
    if (!b) return NULL;
    if (m < 0 || (size_t)m >= b->count || part < 0 || part >= b->parts || i < 0 || j < 0 || (size_t)i >= b->n || (size_t)j >= b->n) {
        PyErr_SetString(PyExc_IndexError, "matrix_exp_ends");
        return NULL;
    }
    if (!b->ok[m]) return Py_NewRef(Py_None);
    dyad lo, hi;
    dyad_init(&lo);
    dyad_init(&hi);
    PyObject *out = pair_of_dyads(&lo, &hi, imat_ends(&lo, &hi, &b->out[m], part, (size_t)i, (size_t)j));
    dyad_free(&lo);
    dyad_free(&hi);
    return out;
}

/* A boundary's setting from attention_settings' item (layout, rounding, ulps numerator, denominator, flush), or None for
   an absent one; -1 with a Python error on failure. */
static int round_item_load(PyObject *item, round_to *r) {
    PyObject *layout;
    long long num, den;
    if (item == Py_None) return 0;
    if (!PyArg_ParseTuple(item, "OiLLp", &layout, &r->rnd, &num, &den, &r->flush) || fmt_load(layout, &r->f)) return -1;
    r->active = 1;
    r->ulps_num = num;
    r->ulps_den = den;
    return 0;
}

/* An accumulation's setting from attention_settings' item (kind, layout, rounding, x, y), x and y the mma's step and
   (reduction, slack), None for any order; None for an absent one; -1 with a Python error on failure. */
static int acc_item_load(PyObject *item, att_acc_set *a) {
    PyObject *layout, *x, *y, *reduction;
    long long products, fraction_bits, exponent_floor, split, count;
    if (item == Py_None) return 0;
    if (!PyArg_ParseTuple(item, "iOiOO", &a->kind, &layout, &a->rnd, &x, &y) || fmt_load(layout, &a->f)) return -1;
    if (a->kind == ATT_SA_MMA) {
        /* x: (products, fraction_bits, exponent_floor, split, count); y: (reduction layout or None, slack (m, e)) */
        PyObject *slack;
        if (!PyArg_ParseTuple(x, "LLLLL", &products, &fraction_bits, &exponent_floor, &split, &count) || !PyArg_ParseTuple(y, "OO", &reduction, &slack) ||
            dyad_load(slack, &a->slack) || (reduction != Py_None && fmt_load(reduction, &a->reduction)))
            return -1;
        a->products = products;
        a->fraction_bits = fraction_bits;
        a->exponent_floor = exponent_floor;
        a->split = split;
        a->count = count;
        a->has_reduction = reduction != Py_None;
    }
    return 0;
}

static PyObject *py_attention_settings(PyObject *self, PyObject *args) {
    PyObject *rounds, *accs, *scale, *methods;
    int flush, backward;
    long long emin, w;
    Py_ssize_t S, E;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOpLLnnOOp", &rounds, &accs, &flush, &emin, &w, &S, &E, &scale, &methods, &backward)) return NULL;
    if (!PyTuple_Check(rounds) || PyTuple_Size(rounds) != ATT_SR_COUNT || !PyTuple_Check(accs) || PyTuple_Size(accs) != ATT_SC_COUNT) {
        PyErr_SetString(PyExc_ValueError, "attention_settings: the boundaries and accumulations");
        return NULL;
    }
    att_settings *s = malloc(sizeof *s);
    if (!s) return PyErr_NoMemory();
    att_settings_init(s);
    s->flush = flush;
    s->emin = emin;
    s->w = w;
    s->S = S;
    s->E = E;
    s->backward = backward;
    for (int z = 0; z < ATT_SR_COUNT; z++)
        if (round_item_load(PyTuple_GetItem(rounds, z), &s->r[z])) goto fail;
    for (int z = 0; z < ATT_SC_COUNT; z++)
        if (acc_item_load(PyTuple_GetItem(accs, z), &s->a[z])) goto fail;
    if (sigma_load(scale, &s->num, &s->den)) goto fail;
    if (methods != Py_None) {
        PyObject *weights;
        long long block_size, split, query_block, key_block;
        if (!PyArg_ParseTuple(methods, "ppppLLOpppppLL", &s->online, &s->exp2, &s->reciprocal, &s->fast_log, &block_size, &split, &weights, &s->has_bias,
                              &s->has_mma, &s->rounded_p, &s->scale_ds, &s->grouped, &query_block, &key_block) ||
            (weights != Py_None && fmt_load(weights, &s->weights)))
            goto fail;
        s->query_block = query_block;
        s->key_block = key_block;
        s->has_methods = 1;
        s->block_size = block_size;
        s->split = split;
        s->has_weights = weights != Py_None;
    }
    {
        int rc;
        Py_BEGIN_ALLOW_THREADS
        rc = att_settings_prepare(s);
        Py_END_ALLOW_THREADS
        if (rc) {
            error(rc);
            goto fail;
        }
    }
    PyObject *capsule = PyCapsule_New(s, ATT_SETTINGS_CAPSULE, att_settings_destroy);
    if (capsule) return capsule;
fail:
    att_settings_free(s);
    free(s);
    return NULL;
}

/* lemma1_ends(round item, flush, lo, hi, w): Lemma 1 at a boundary (errors.c's lean_of and lean_ends) on [lo, hi],
   the setting as attention_settings' round items: the interval held, or None where the rounding may overflow (for the
   tests). */
static PyObject *py_lemma1_ends(PyObject *self, PyObject *args) {
    PyObject *item, *loo, *hio;
    int flush;
    long long w;
    (void)self;
    if (!PyArg_ParseTuple(args, "OpOOL", &item, &flush, &loo, &hio, &w)) return NULL;
    round_to r;
    memset(&r, 0, sizeof r);
    round_bound l;
    lean_init(&l);
    dyad lo, hi;
    dyad_init(&lo);
    dyad_init(&hi);
    PyObject *out = NULL;
    if (!round_item_load(item, &r) && !dyad_load(loo, &lo) && !dyad_load(hio, &hi)) {
        int unbounded = 0, rc = lean_of(&l, &r, flush);
        if (!rc) rc = lean_ends(&lo, &hi, &unbounded, &l, w);
        out = rc ? error(rc) : unbounded ? Py_NewRef(Py_None) : pair_of_dyads(&lo, &hi, 0);
    }
    lean_free(&l);
    dyad_free(&lo);
    dyad_free(&hi);
    return out;
}

/* accumulation_error(accumulation item, n, mass, w, flush): the bound on n terms' accumulation of total magnitude mass
   (attention_tol.c's acc_of and acc_apply), the setting as attention_settings' accumulation items: the error, or None where
   the sum may overflow (for the tests). */
static PyObject *py_accumulation_error(PyObject *self, PyObject *args) {
    PyObject *item, *mo;
    long long n, w;
    int flush;
    (void)self;
    if (!PyArg_ParseTuple(args, "OLOLp", &item, &n, &mo, &w, &flush)) return NULL;
    att_acc_set a;
    memset(&a, 0, sizeof a);
    dyad_init(&a.slack);
    att_acc c;
    memset(&c, 0, sizeof c);
    ivl x;
    ivl_init(&x);
    dyad mass;
    dyad_init(&mass);
    PyObject *out = NULL;
    if (!acc_item_load(item, &a) && !dyad_load(mo, &mass)) {
        int unbounded = 0, rc = n < 0 ? EXACTREF_INTERNAL : acc_of(&c, &a, n, n, w, flush);
        if (!rc) rc = acc_apply(&x, &c, (size_t)n, &mass, w, &unbounded);
        out = rc ? error(rc) : unbounded ? Py_NewRef(Py_None) : dyad_store(&x.hi, 0);
    }
    if (c.terms) {
        for (size_t z = 0; z < 3 * (c.nmax + 1); z++) dyad_free(&c.terms[z]);
        free(c.terms);
    }
    free(c.has);
    dyad_free(&a.slack);
    ivl_free(&x);
    dyad_free(&mass);
    return out;
}

/* ivl_op(op, (lo, hi), (lo, hi), p): bigfloat.c's interval operations, op one of "add", "sub", "mul", "div" (the second
   positive), "max", "abs" (of the first): the result's ends (for the tests). */
static PyObject *py_ivl_op(PyObject *self, PyObject *args) {
    const char *op;
    PyObject *ao, *bo;
    long long p;
    (void)self;
    if (!PyArg_ParseTuple(args, "sOOL", &op, &ao, &bo, &p)) return NULL;
    PyObject *a0, *a1, *b0, *b1;
    if (!PyArg_ParseTuple(ao, "OO", &a0, &a1) || !PyArg_ParseTuple(bo, "OO", &b0, &b1)) return NULL;
    ivl a, b, r;
    ivl_init(&a);
    ivl_init(&b);
    ivl_init(&r);
    PyObject *out = NULL;
    if (!dyad_load(a0, &a.lo) && !dyad_load(a1, &a.hi) && !dyad_load(b0, &b.lo) && !dyad_load(b1, &b.hi)) {
        int rc = !strcmp(op, "add") ? ivl_add(&r, &a, &b, p) : !strcmp(op, "sub") ? ivl_sub(&r, &a, &b, p) : !strcmp(op, "mul") ? ivl_mul(&r, &a, &b, p)
               : !strcmp(op, "div") ? ivl_div(&r, &a, &b, p) : !strcmp(op, "max") ? ivl_max(&r, &a, &b) : !strcmp(op, "abs") ? ivl_abs(&r, &a)
               : EXACTREF_DOMAIN;
        out = pair_of_dyads(&r.lo, &r.hi, rc);
    }
    ivl_free(&a);
    ivl_free(&b);
    ivl_free(&r);
    return out;
}

/* Lemma C (att_carry_add) for the tests: blocks carried in the format of round (attention_settings' boundary item), each
   summed on accumulation's (its accumulations item, for a block's terms + 2), the scale (m, e) or None, and blocks a
   sequence of (lo, hi, mass, factors, e, count): a block's exact sums, mass and (scaled) factors' magnitudes as integers
   at 2**e, and its count of terms that can be nonzero. The enclosure ((m, e), (m, e)), or None without a bound. */
static PyObject *py_attention_carry(PyObject *self, PyObject *args) {
    PyObject *ritem, *aitem, *scale, *blocks, *out = NULL;
    int flush;
    long long w;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOpLO", &ritem, &aitem, &scale, &flush, &w, &blocks)) return NULL;
    PyObject *seq = PySequence_Tuple(blocks);
    if (!seq) return NULL;
    Py_ssize_t nb = PyTuple_Size(seq);
    round_to r;
    att_acc_set a;
    att_carry k;
    memset(&r, 0, sizeof r);
    memset(&a, 0, sizeof a);
    dyad_init(&a.slack);
    att_carry_init(&k);
    dyad sigma, c[2], lo, hi, mass, F;
    dyad *ds[] = {&sigma, &c[0], &c[1], &lo, &hi, &mass, &F};
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_init(ds[z]);
    int scaled = scale != Py_None, rc = 0;
    unsigned char status = 0;
    long long last = 0;
    if (round_item_load(ritem, &r) || acc_item_load(aitem, &a) || (scaled && dyad_load(scale, &sigma))) goto done;
    for (Py_ssize_t b = 0; b < nb; b++) {
        long long e, count;
        PyObject *item = PyTuple_GetItem(seq, b), *x[4];
        if (!PyArg_ParseTuple(item, "OOOOLL", &x[0], &x[1], &x[2], &x[3], &e, &count)) goto done;
        if (count + 2 > last) last = count + 2;
    }
    Py_BEGIN_ALLOW_THREADS
    rc = att_carry_build(&k, &r, &a, last, scaled, &sigma, &sigma, flush, w);
    Py_END_ALLOW_THREADS
    if (rc) {
        error(rc);
        goto done;
    }
    for (Py_ssize_t b = 0; b < nb && !rc && status != 2; b++) {
        long long e, count;
        PyObject *x[4];
        if (!PyArg_ParseTuple(PyTuple_GetItem(seq, b), "OOOOLL", &x[0], &x[1], &x[2], &x[3], &e, &count) || int_load(x[0], &lo.m) ||
            int_load(x[1], &hi.m) || int_load(x[2], &mass.m) || (scaled && int_load(x[3], &F.m)))
            goto done;
        lo.e = lo.m.n ? e : 0; /* a zero is 0 2**0 */
        hi.e = hi.m.n ? e : 0;
        mass.e = mass.m.n ? e : 0;
        F.e = F.m.n ? e : 0;
        rc = att_carry_add(c, &status, &k, scaled, &lo, &hi, &mass, count, &F);
    }
    if (rc)
        error(rc);
    else if (status == 2)
        out = Py_NewRef(Py_None);
    else
        out = pair_of_dyads(&c[0], &c[1], 0);
done:
    Py_DECREF(seq);
    dyad_free(&a.slack);
    att_carry_free(&k);
    for (size_t z = 0; z < sizeof ds / sizeof *ds; z++) dyad_free(ds[z]);
    return out;
}

/* EXACTREF_NEAREST as _rounding.nearest_accumulator's ValueError, else error(rc) */
static PyObject *settings_error(int rc, int bad) {
    static const char *names[] = {"NEAREST_EVEN", "NEAREST_AWAY", "TOWARD_ZERO", "TOWARD_POSITIVE", "TOWARD_NEGATIVE"};
    if (rc != EXACTREF_NEAREST) return error(rc);
    PyErr_Format(PyExc_ValueError, "an accumulator rounds to nearest (NEAREST_EVEN or NEAREST_AWAY), not %s", bad >= 0 && bad < 5 ? names[bad] : "?");
    return NULL;
}

static PyObject *py_attention_tolerance_build(PyObject *self, PyObject *args) {
    PyObject *so, *f64_layout;
    int normal;
    (void)self;
    if (!PyArg_ParseTuple(args, "OpO", &so, &normal, &f64_layout)) return NULL;
    const att_settings *s = PyCapsule_GetPointer(so, ATT_SETTINGS_CAPSULE);
    if (!s) return NULL;
    att_tol *t = malloc(sizeof *t);
    if (!t) return PyErr_NoMemory();
    att_tol_init(t);
    att_call a;
    memset(&a, 0, sizeof a);
    big_init(&a.num);
    big_init(&a.den);
    int applicable = 0, bad = -1, rc = fmt_load(f64_layout, &t->f64) ? -1 : 0;
    if (!rc && !(rc = big_copy(&a.num, &s->num)) && !(rc = big_copy(&a.den, &s->den))) {
        a.E = s->E;
        Py_BEGIN_ALLOW_THREADS
        rc = att_tol_build(t, &applicable, &bad, s, normal);
        if (!rc && applicable) rc = att_tol_prepare(t, &a, &s->a[ATT_SC_SCORES].f, s->w);
        Py_END_ALLOW_THREADS
    }
    big_free(&a.num);
    big_free(&a.den);
    PyObject *out = NULL;
    if (!rc && applicable && (out = PyCapsule_New(t, ATT_TOL_CAPSULE, att_tol_destroy))) return out;
    att_tol_free(t);
    free(t);
    if (rc > 0) return settings_error(rc, bad);
    if (rc || PyErr_Occurred()) return NULL; /* PyCapsule_New's failure set its exception */
    return Py_NewRef(Py_None);
}

#define ATT_GTOL_CAPSULE "exactref._c.att_gtol"

static void att_gtol_destroy(PyObject *capsule) {
    att_gtol *g = PyCapsule_GetPointer(capsule, ATT_GTOL_CAPSULE);
    if (g) {
        att_gtol_free(g);
        free(g);
    }
}

static PyObject *py_attention_gtol_build(PyObject *self, PyObject *args) {
    PyObject *so, *f64_layout;
    int normal, closed, sign;
    Py_ssize_t Ev, nq, nkv, heads;
    (void)self;
    if (!PyArg_ParseTuple(args, "OppiOnnnn", &so, &normal, &closed, &sign, &f64_layout, &Ev, &nq, &nkv, &heads)) return NULL;
    const att_settings *s = PyCapsule_GetPointer(so, ATT_SETTINGS_CAPSULE);
    if (!s) return NULL;
    att_gtol *g = malloc(sizeof *g);
    if (!g) return PyErr_NoMemory();
    att_gtol_init(g);
    att_call a;
    memset(&a, 0, sizeof a);
    big_init(&a.num);
    big_init(&a.den);
    int bad = -1, rc = fmt_load(f64_layout, &g->t.f64) || fmt_load(f64_layout, &g->g.f64) ? -1 : 0;
    if (!rc && !(rc = big_copy(&a.num, &s->num)) && !(rc = big_copy(&a.den, &s->den))) {
        a.E = s->E;
        Py_BEGIN_ALLOW_THREADS
        rc = att_gtol_build(g, &bad, s, &a, normal, closed, sign, Ev, nq, nkv, heads);
        Py_END_ALLOW_THREADS
    }
    big_free(&a.num);
    big_free(&a.den);
    PyObject *out = NULL;
    if (!rc && (out = PyCapsule_New(g, ATT_GTOL_CAPSULE, att_gtol_destroy))) return out;
    att_gtol_free(g);
    free(g);
    if (rc > 0) return settings_error(rc, bad);
    return NULL;
}

#define ATT_GTOL_CALL_CAPSULE "exactref._c.att_gtol_call"
#define ATT_GTOL_GROUP_CAPSULE "exactref._c.att_gtol_group"

typedef struct {
    att_gtol_call c;
    PyObject *keep; /* the Python objects whose data the call reads */
} grad_call_box;

typedef struct {
    att_gtol_group g;
    PyObject *call, *kb, *vb; /* its call's capsule and its blocks' */
} grad_group_box;

static void grad_call_destroy(PyObject *capsule) {
    grad_call_box *b = PyCapsule_GetPointer(capsule, ATT_GTOL_CALL_CAPSULE);
    if (!b) return;
    att_gtol_call_free(&b->c);
    Py_XDECREF(b->keep);
    free(b);
}

static void grad_group_destroy(PyObject *capsule) {
    grad_group_box *b = PyCapsule_GetPointer(capsule, ATT_GTOL_GROUP_CAPSULE);
    if (!b) return;
    att_gtol_group_free(&b->g);
    Py_XDECREF(b->call);
    Py_XDECREF(b->kb);
    Py_XDECREF(b->vb);
    free(b);
}

#define ATT_GREF_CALL_CAPSULE "exactref._c.att_gref_call"
#define ATT_GREF_GROUP_CAPSULE "exactref._c.att_gref_group"

typedef struct {
    att_gref_call c;
    PyObject *keep; /* the Python objects whose data the call reads */
} gref_call_box;

typedef struct {
    att_gref_group g;
    PyObject *call, *kb, *vb; /* its call's capsule and its blocks' */
} gref_group_box;

static void gref_call_destroy(PyObject *capsule) {
    gref_call_box *b = PyCapsule_GetPointer(capsule, ATT_GREF_CALL_CAPSULE);
    if (!b) return;
    att_gref_call_free(&b->c);
    Py_XDECREF(b->keep);
    free(b);
}

static void gref_group_destroy(PyObject *capsule) {
    gref_group_box *b = PyCapsule_GetPointer(capsule, ATT_GREF_GROUP_CAPSULE);
    if (!b) return;
    att_gref_group_free(&b->g);
    Py_XDECREF(b->call);
    Py_XDECREF(b->kb);
    Py_XDECREF(b->vb);
    free(b);
}

/* attention_gref_refine(call, rows, elems, rowlist, p0, cap): the backward's refinement (attention_grad_refine.c) of the
   elements its reference left undecided, as a call: rows as int64 (query row, query index, mask offset, mask step,
   output-gradient row, key block row, value block row), elements as int64 (kind, key, column, first, count into
   rowlist). attention_gref_refine_next gives the tiles of the next wave of rows (0 when done), attention_gref_refine_tile
   runs one (off the GIL, on the workers), attention_gref_refine_result gives per element its encoding, entry flag, entry
   midpoint and status (1: undecided at the cap). */
#define ATT_GREF_RUN_CAPSULE "exactref._c.att_gref_run"

typedef struct {
    att_gref_refine f;
    PyObject *call;           /* the call capsule the refinement reads */
    PyObject *res[4];         /* encodings, entry flags, entry midpoints, statuses */
    entry *entries;
    int done;
} gref_run_box;

static void gref_run_free(gref_run_box *b) {
    att_gref_refine_free(&b->f);
    free(b->entries);
    Py_XDECREF(b->call);
    for (int k = 0; k < 4; k++) Py_XDECREF(b->res[k]);
    free(b);
}

static void gref_run_destroy(PyObject *capsule) {
    gref_run_box *b = PyCapsule_GetPointer(capsule, ATT_GREF_RUN_CAPSULE);
    if (b) gref_run_free(b);
}

static PyObject *py_attention_gref_refine(PyObject *self, PyObject *args) {
    PyObject *capsule, *rows_obj, *elems_obj, *list_obj;
    long long p0, cap;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOLL", &capsule, &rows_obj, &elems_obj, &list_obj, &p0, &cap)) return NULL;
    gref_call_box *c = PyCapsule_GetPointer(capsule, ATT_GREF_CALL_CAPSULE);
    if (!c) return NULL;
    const unsigned char *rows, *elems, *list;
    Py_ssize_t rlen, elen, llen;
    if (data_load(rows_obj, &rows, &rlen) || data_load(elems_obj, &elems, &elen) || data_load(list_obj, &list, &llen)) return NULL;
    size_t nr = (size_t)rlen / 56, ne = (size_t)elen / 40, nl = (size_t)llen / 8, size = c->c.out_size;
    gref_run_box *b = calloc(1, sizeof *b);
    if (!b) return PyErr_NoMemory();
    att_gref_refine_init(&b->f);
    PyObject *out = NULL;
    Py_ssize_t sizes[4] = {(Py_ssize_t)(ne * size), (Py_ssize_t)ne, (Py_ssize_t)(ne * ENTRY_WIDTH), (Py_ssize_t)ne};
    for (int k = 0; k < 4; k++)
        if (!(b->res[k] = PyBytes_FromStringAndSize(NULL, sizes[k]))) goto done;
    unsigned char *o[4];
    for (int k = 0; k < 4; k++) o[k] = (unsigned char *)PyBytes_AsString(b->res[k]);
    memset(o[0], 0, (size_t)sizes[0]);
    int rc = 0;
    att_gref_refine *f = &b->f;
    f->c = &c->c;
    f->nrows = nr;
    f->count = ne;
    f->rows = calloc(nr + 1, sizeof *f->rows);
    f->elems = calloc(ne + 1, sizeof *f->elems);
    f->rowlist = calloc(nl + 1, sizeof *f->rowlist);
    b->entries = calloc(ne + 1, sizeof *b->entries);
    f->entries = b->entries;
    f->enc = o[0];
    f->status = o[3];
    if (!f->rows || !f->elems || !f->rowlist || !f->entries) rc = EXACTREF_NOMEM;
    for (size_t r = 0; !rc && r < nr; r++) {
        int64_t v[7];
        memcpy(v, rows + 56 * r, sizeof v);
        att_grow *g = &f->rows[r];
        g->qr = (size_t)v[0], g->i = (size_t)v[1], g->moff = (size_t)v[2], g->mstep = (size_t)v[3], g->gr0 = (size_t)v[4], g->kr0 = (size_t)v[5], g->vr0 = (size_t)v[6];
    }
    for (size_t e = 0; !rc && e < ne; e++) {
        int64_t v[5];
        memcpy(v, elems + 40 * e, sizeof v);
        att_gelem *el = &f->elems[e];
        el->kind = (int)v[0], el->j = (size_t)v[1], el->col = (size_t)v[2], el->first = (size_t)v[3], el->count = (size_t)v[4];
        if (el->first + el->count > nl) rc = EXACTREF_INTERNAL;
        o[3][e] = 1;
    }
    for (size_t k = 0; !rc && k < nl; k++) {
        int64_t v;
        memcpy(&v, list + 8 * k, 8);
        if ((size_t)v >= nr) rc = EXACTREF_INTERNAL;
        f->rowlist[k] = (size_t)v;
    }
    if (!rc) rc = att_gref_refine_start(f, p0, cap);
    if (rc) {
        error(rc);
        goto done;
    }
    Py_INCREF(capsule);
    b->call = capsule;
    if ((out = PyCapsule_New(b, ATT_GREF_RUN_CAPSULE, gref_run_destroy))) b = NULL;
done:
    if (b) gref_run_free(b);
    return out;
}

/* attention_gref_refine_next(run): the tiles of the next wave (0 when done) */
static PyObject *py_attention_gref_refine_next(PyObject *self, PyObject *args) {
    PyObject *capsule;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    gref_run_box *b = PyCapsule_GetPointer(capsule, ATT_GREF_RUN_CAPSULE);
    if (!b) return NULL;
    size_t tiles = 0;
    int rc = 0;
    if (!b->done) {
        Py_BEGIN_ALLOW_THREADS
        rc = att_gref_refine_next(&b->f, &tiles);
        Py_END_ALLOW_THREADS
        b->done = !rc && !tiles;
    }
    if (rc) return error(rc);
    return PyLong_FromSize_t(tiles);
}

/* attention_gref_refine_waves(run): (its waves so far, those that ran split), for tests */
static PyObject *py_attention_gref_refine_waves(PyObject *self, PyObject *args) {
    PyObject *capsule;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    gref_run_box *b = PyCapsule_GetPointer(capsule, ATT_GREF_RUN_CAPSULE);
    if (!b) return NULL;
    size_t waves, split;
    att_gref_refine_waves(&b->f, &waves, &split);
    return Py_BuildValue("(nn)", (Py_ssize_t)waves, (Py_ssize_t)split);
}

/* attention_gref_refine_tile(run, t): tile t of the current wave */
static PyObject *py_attention_gref_refine_tile(PyObject *self, PyObject *args) {
    PyObject *capsule;
    Py_ssize_t t;
    (void)self;
    if (!PyArg_ParseTuple(args, "On", &capsule, &t)) return NULL;
    gref_run_box *b = PyCapsule_GetPointer(capsule, ATT_GREF_RUN_CAPSULE);
    if (!b) return NULL;
    if (b->done || t < 0) {
        PyErr_SetString(PyExc_ValueError, "attention_gref_refine_tile: no such tile");
        return NULL;
    }
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = att_gref_refine_tile(&b->f, (size_t)t);
    Py_END_ALLOW_THREADS
    if (rc) return error(rc);
    return Py_NewRef(Py_None);
}

/* attention_gref_refine_result(run): (encodings, entry flags, entry midpoints, statuses) once it is done */
static PyObject *py_attention_gref_refine_result(PyObject *self, PyObject *args) {
    PyObject *capsule;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    gref_run_box *b = PyCapsule_GetPointer(capsule, ATT_GREF_RUN_CAPSULE);
    if (!b) return NULL;
    if (!b->done) {
        PyErr_SetString(PyExc_ValueError, "attention_gref_refine_result: the refinement is not done");
        return NULL;
    }
    unsigned char *flags = (unsigned char *)PyBytes_AsString(b->res[1]), *mids = (unsigned char *)PyBytes_AsString(b->res[2]);
    for (size_t e = 0; e < b->f.count; e++) {
        flags[e] = b->entries[e].flag;
        memcpy(mids + e * ENTRY_WIDTH, b->entries[e].mid, ENTRY_WIDTH);
    }
    return PyTuple_Pack(4, b->res[0], b->res[1], b->res[2], b->res[3]);
}

/* attention_gref_call(q, k, gd, vd, gs, qs, ks, tables, scale, E, u, S, Ev, causal, mask, mask layout, out layout, rounding,
   overflow, p, nq, nkv, query slots): a call of the backward's reference (attention_gref.c), as a capsule */
static PyObject *py_attention_gref_call(PyObject *self, PyObject *args) {
    PyObject *qo, *ko, *gdo, *vdo, *gso, *qso, *kso, *to, *sig, *mask_obj, *mask_layout, *layout;
    Py_ssize_t E, u, S, Ev, p, nq, nkv, slots;
    int causal, rnd, ovf;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOnnnnpOOOiinnnn", &qo, &ko, &gdo, &vdo, &gso, &qso, &kso, &to, &sig, &E, &u, &S, &Ev, &causal, &mask_obj,
                          &mask_layout, &layout, &rnd, &ovf, &p, &nq, &nkv, &slots))
        return NULL;
    gref_call_box *b = calloc(1, sizeof *b);
    if (!b) return PyErr_NoMemory();
    att_gref_call *c = &b->c;
    att_gref_call_init(c);
    PyObject *out = NULL;
    partfmt pf = {0};
    c->a.q = operand_get(qo);
    c->a.k = operand_get(ko);
    c->gd = operand_get(gdo);
    c->vd = operand_get(vdo);
    c->gs = operand_get(gso);
    c->qs = operand_get(qso);
    c->ks = operand_get(kso);
    c->a.tables = to == Py_None ? NULL : PyCapsule_GetPointer(to, ATT_TABLES_CAPSULE); /* None: Lemma X per weight */
    const unsigned char *mask_data = NULL;
    Py_ssize_t mask_len;
    if (!c->a.q || !c->a.k || !c->gd || !c->vd || !c->gs || !c->qs || !c->ks || (to != Py_None && !c->a.tables) || sigma_load(sig, &c->a.num, &c->a.den) ||
        (mask_obj != Py_None && data_load(mask_obj, &mask_data, &mask_len)) || partfmt_load(layout, &pf, c->out_nan))
        goto done;
    if (mask_obj != Py_None) c->a.mask = mask_data;
    if (mask_load(mask_layout, &c->a)) goto done;
    c->a.mtables = c->a.tables ? c->a.tables + 1 : NULL;
    c->a.S = (size_t)S;
    c->a.Ev = (size_t)Ev;
    c->a.causal = causal;
    c->a.sign = !c->a.den.n ? 1 : !c->a.num.n ? 0 : c->a.num.neg ? -1 : 1;
    c->a.E = E;
    c->a.u = u;
    c->a.p = p;
    c->p = p;
    c->out = pf.f;
    c->out_size = part_size(&pf);
    c->rnd = rnd;
    c->ovf = ovf;
    c->neg_zero = rnd == ROUND_TOWARD_NEGATIVE;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = att_gref_call_prepare(c, nq, nkv);
    if (!rc) rc = att_gref_query_slots(c, (size_t)slots);
    Py_END_ALLOW_THREADS
    if (rc) {
        error(rc);
        goto done;
    }
    if (!(b->keep = PyTuple_Pack(9, qo, ko, gdo, vdo, gso, qso, kso, to, mask_obj))) goto done;
    out = PyCapsule_New(b, ATT_GREF_CALL_CAPSULE, gref_call_destroy);
done:
    if (!out) {
        att_gref_call_free(c);
        Py_XDECREF(b->keep);
        free(b);
    }
    return out;
}

#define ATT_GREF_BLOCK_CAPSULE "exactref._c.att_gref_block"

static void gref_block_destroy(PyObject *capsule) {
    att_gref_block *b = PyCapsule_GetPointer(capsule, ATT_GREF_BLOCK_CAPSULE);
    if (!b) return;
    att_gref_block_free(b);
    free(b);
}

/* attention_gref_block(call, key): a key block's grad_key elements (key) or a value block's grad_value elements, as a
   capsule */
static PyObject *py_attention_gref_block(PyObject *self, PyObject *args) {
    PyObject *co;
    int key;
    (void)self;
    if (!PyArg_ParseTuple(args, "Op", &co, &key)) return NULL;
    gref_call_box *cb = PyCapsule_GetPointer(co, ATT_GREF_CALL_CAPSULE);
    if (!cb) return NULL;
    att_gref_block *b = malloc(sizeof *b);
    if (!b) return PyErr_NoMemory();
    int rc = att_gref_block_start(b, &cb->c, key);
    PyObject *out = rc ? error(rc) : PyCapsule_New(b, ATT_GREF_BLOCK_CAPSULE, gref_block_destroy);
    if (!out) {
        att_gref_block_free(b);
        free(b);
    }
    return out;
}

/* attention_gref_group(call, kr0, vr0, kblock, vblock, cap): a key and value unit's tile, adding to the two blocks */
static PyObject *py_attention_gref_group(PyObject *self, PyObject *args) {
    PyObject *co, *ko, *vo;
    Py_ssize_t kr0, vr0, cap;
    (void)self;
    if (!PyArg_ParseTuple(args, "OnnOOn", &co, &kr0, &vr0, &ko, &vo, &cap)) return NULL;
    gref_call_box *cb = PyCapsule_GetPointer(co, ATT_GREF_CALL_CAPSULE);
    att_gref_block *kb = cb ? PyCapsule_GetPointer(ko, ATT_GREF_BLOCK_CAPSULE) : NULL, *vb = kb ? PyCapsule_GetPointer(vo, ATT_GREF_BLOCK_CAPSULE) : NULL;
    if (!vb) return NULL;
    gref_group_box *b = calloc(1, sizeof *b);
    if (!b) return PyErr_NoMemory();
    att_gref_group_init(&b->g);
    int rc = att_gref_group_start(&b->g, &cb->c, (size_t)kr0, (size_t)vr0, kb, vb, (size_t)cap);
    PyObject *out = rc ? error(rc) : PyCapsule_New(b, ATT_GREF_GROUP_CAPSULE, gref_group_destroy);
    if (!out) {
        att_gref_group_free(&b->g);
        free(b);
        return NULL;
    }
    b->call = Py_NewRef(co);
    b->kb = Py_NewRef(ko);
    b->vb = Py_NewRef(vo);
    return out;
}

static int gref_args(PyObject *go, PyObject *co, gref_group_box **g, gref_call_box **c) {
    *g = PyCapsule_GetPointer(go, ATT_GREF_GROUP_CAPSULE);
    *c = *g && co ? PyCapsule_GetPointer(co, ATT_GREF_CALL_CAPSULE) : NULL;
    return *g && (!co || *c) ? 0 : -1;
}

/* attention_gref_tile(group, rows): the group's next tile, its rows as attention_gtol_tile's */
static PyObject *py_attention_gref_tile(PyObject *self, PyObject *args) {
    PyObject *go, *rows_obj;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &go, &rows_obj)) return NULL;
    gref_group_box *b = PyCapsule_GetPointer(go, ATT_GREF_GROUP_CAPSULE);
    const unsigned char *rows;
    Py_ssize_t len;
    if (!b || data_load(rows_obj, &rows, &len)) return NULL;
    size_t T = (size_t)len / (8 * ATT_GR_INFO);
    if (T > b->g.cap) {
        PyErr_SetString(PyExc_ValueError, "attention_gref_tile: more rows than the group's tiles hold");
        return NULL;
    }
    memcpy(b->g.rinfo, rows, T * 8 * ATT_GR_INFO);
    b->g.T = T;
    return Py_NewRef(Py_None);
}

/* a phase on the group: rows (r0, r1), the tile's denominators, keys (j0, j1), or the tile's end */
static PyObject *gref_phase(PyObject *args, int which) {
    PyObject *go, *co = NULL;
    Py_ssize_t x0 = 0, x1 = 0;
    if (which == 3 ? !PyArg_ParseTuple(args, "O", &go) : which == 1 ? !PyArg_ParseTuple(args, "OO", &go, &co) : !PyArg_ParseTuple(args, "OOnn", &go, &co, &x0, &x1))
        return NULL;
    gref_group_box *g;
    gref_call_box *c;
    if (gref_args(go, co, &g, &c)) return NULL;
    int rc = 0;
    Py_BEGIN_ALLOW_THREADS
    if (which == 0) rc = att_gref_tile_rows(&g->g, &c->c, (size_t)x0, (size_t)x1);
    else if (which == 1) rc = att_gref_tile_prep(&g->g, &c->c);
    else if (which == 2) rc = att_gref_tile_keys(&g->g, &c->c, (size_t)x0, (size_t)x1);
    else att_gref_tile_done(&g->g);
    Py_END_ALLOW_THREADS
    if (rc) return error(rc);
    return Py_NewRef(Py_None);
}

static PyObject *py_attention_gref_tile_rows(PyObject *self, PyObject *args) {
    (void)self;
    return gref_phase(args, 0);
}

static PyObject *py_attention_gref_tile_prep(PyObject *self, PyObject *args) {
    (void)self;
    return gref_phase(args, 1);
}

static PyObject *py_attention_gref_tile_keys(PyObject *self, PyObject *args) {
    (void)self;
    return gref_phase(args, 2);
}

static PyObject *py_attention_gref_tile_done(PyObject *self, PyObject *args) {
    (void)self;
    return gref_phase(args, 3);
}

/* n elements' encodings, entry flags, entry midpoints and statuses as bytes */
static PyObject *gref_outputs(const unsigned char *enc, const entry *x, const unsigned char *status, size_t n, size_t size) {
    PyObject *e = PyBytes_FromStringAndSize((const char *)enc, (Py_ssize_t)(n * size)), *f = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)n),
             *m = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(n * ENTRY_WIDTH)), *s = PyBytes_FromStringAndSize((const char *)status, (Py_ssize_t)n);
    PyObject *out = NULL;
    if (e && f && m && s) {
        unsigned char *fd = (unsigned char *)PyBytes_AsString(f), *md = (unsigned char *)PyBytes_AsString(m);
        for (size_t i = 0; i < n; i++) {
            fd[i] = x[i].flag;
            memcpy(md + i * ENTRY_WIDTH, x[i].mid, ENTRY_WIDTH);
        }
        out = PyTuple_Pack(4, e, f, m, s);
    }
    Py_XDECREF(e);
    Py_XDECREF(f);
    Py_XDECREF(m);
    Py_XDECREF(s);
    return out;
}

/* attention_gref_tile_queries(group, call, r0, r1): rows r0 to r1's grad_query (E elements per row) as gref_outputs */
static PyObject *py_attention_gref_tile_queries(PyObject *self, PyObject *args) {
    PyObject *go, *co;
    Py_ssize_t r0, r1;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOnn", &go, &co, &r0, &r1)) return NULL;
    gref_group_box *g;
    gref_call_box *c;
    if (gref_args(go, co, &g, &c)) return NULL;
    size_t n = (size_t)(r1 - r0) * (size_t)c->c.a.E, size = c->c.out_size;
    unsigned char *enc = calloc((n ? n : 1) * size, 1), *status = calloc(n ? n : 1, 1);
    entry *x = calloc(n ? n : 1, sizeof *x);
    PyObject *out = NULL;
    int rc = enc && status && x ? 0 : EXACTREF_NOMEM;
    if (!rc) {
        Py_BEGIN_ALLOW_THREADS
        rc = att_gref_tile_queries(&g->g, &c->c, (size_t)r0, (size_t)r1, enc, x, status);
        Py_END_ALLOW_THREADS
    }
    out = rc ? error(rc) : gref_outputs(enc, x, status, n, size);
    free(enc);
    free(status);
    free(x);
    return out;
}

/* attention_gref_block_finish(block, call, key, j0, j1): keys j0 to j1's elements after the block's last unit, as gref_outputs */
static PyObject *py_attention_gref_block_finish(PyObject *self, PyObject *args) {
    PyObject *bo, *co;
    int key;
    Py_ssize_t j0, j1;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOpnn", &bo, &co, &key, &j0, &j1)) return NULL;
    att_gref_block *b = PyCapsule_GetPointer(bo, ATT_GREF_BLOCK_CAPSULE);
    gref_call_box *cb = b ? PyCapsule_GetPointer(co, ATT_GREF_CALL_CAPSULE) : NULL;
    if (!cb) return NULL;
    size_t n = (size_t)(j1 - j0) * b->cols, size = cb->c.out_size;
    unsigned char *enc = calloc((n ? n : 1) * size, 1), *status = calloc(n ? n : 1, 1);
    entry *x = calloc(n ? n : 1, sizeof *x);
    PyObject *out = NULL;
    int rc = enc && status && x ? 0 : EXACTREF_NOMEM;
    if (!rc) {
        Py_BEGIN_ALLOW_THREADS
        rc = att_gref_block_finish(b, &cb->c, key, (size_t)j0, (size_t)j1, enc, x, status);
        Py_END_ALLOW_THREADS
    }
    out = rc ? error(rc) : gref_outputs(enc, x, status, n, size);
    free(enc);
    free(status);
    free(x);
    return out;
}

#define ATT_GNORM_CALL_CAPSULE "exactref._c.att_gnorm_call"
#define ATT_GNORM_GROUP_CAPSULE "exactref._c.att_gnorm_group"

typedef struct {
    att_gnorm_call c;
    PyObject *keep; /* the Python objects whose data the call reads */
} gnorm_call_box;

typedef struct {
    att_gnorm_group g;
    PyObject *call, *kb, *vb; /* its call's capsule and its blocks' */
} gnorm_group_box;

static void gnorm_call_destroy(PyObject *capsule) {
    gnorm_call_box *b = PyCapsule_GetPointer(capsule, ATT_GNORM_CALL_CAPSULE);
    if (!b) return;
    att_gnorm_call_free(&b->c);
    Py_XDECREF(b->keep);
    free(b);
}

static void gnorm_group_destroy(PyObject *capsule) {
    gnorm_group_box *b = PyCapsule_GetPointer(capsule, ATT_GNORM_GROUP_CAPSULE);
    if (!b) return;
    att_gnorm_group_free(&b->g);
    Py_XDECREF(b->call);
    Py_XDECREF(b->kb);
    Py_XDECREF(b->vb);
    free(b);
}

#define ATT_GNORM_BLOCK_CAPSULE "exactref._c.att_gnorm_block"

static void gnorm_block_destroy(PyObject *capsule) {
    att_gnorm_block *b = PyCapsule_GetPointer(capsule, ATT_GNORM_BLOCK_CAPSULE);
    if (!b) return;
    att_gnorm_block_free(b);
    free(b);
}

/* attention_gnorm_block(call, cols): a key block's grad_key sums (cols E) or a value block's grad_value sums (cols Ev), as a capsule */
static PyObject *py_attention_gnorm_block(PyObject *self, PyObject *args) {
    PyObject *co;
    Py_ssize_t cols;
    (void)self;
    if (!PyArg_ParseTuple(args, "On", &co, &cols)) return NULL;
    gnorm_call_box *cb = PyCapsule_GetPointer(co, ATT_GNORM_CALL_CAPSULE);
    if (!cb) return NULL;
    att_gnorm_block *b = malloc(sizeof *b);
    if (!b) return PyErr_NoMemory();
    int rc = att_gnorm_block_start(b, &cb->c, (size_t)cols);
    PyObject *out = rc ? error(rc) : PyCapsule_New(b, ATT_GNORM_BLOCK_CAPSULE, gnorm_block_destroy);
    if (!out) {
        att_gnorm_block_free(b);
        free(b);
    }
    return out;
}

/* attention_gnorm_block_finish(block, call, scaled, j0, j1): keys j0 to j1's elements after the block's last unit: (encodings, statuses) */
static PyObject *py_attention_gnorm_block_finish(PyObject *self, PyObject *args) {
    PyObject *bo, *co;
    int scaled;
    Py_ssize_t j0, j1;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOpnn", &bo, &co, &scaled, &j0, &j1)) return NULL;
    att_gnorm_block *b = PyCapsule_GetPointer(bo, ATT_GNORM_BLOCK_CAPSULE);
    gnorm_call_box *cb = b ? PyCapsule_GetPointer(co, ATT_GNORM_CALL_CAPSULE) : NULL;
    if (!cb) return NULL;
    const att_gnorm_call *c = &cb->c;
    size_t n = (size_t)(j1 - j0) * b->cols, size = c->out_size;
    PyObject *e = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(n * size)), *st = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)n), *out = NULL;
    if (e && st) {
        int rc;
        unsigned char *ed = (unsigned char *)PyBytes_AsString(e), *sd = (unsigned char *)PyBytes_AsString(st);
        Py_BEGIN_ALLOW_THREADS
        rc = att_gnorm_block_finish(b, c, scaled, (size_t)j0, (size_t)j1, ed, sd);
        Py_END_ALLOW_THREADS
        out = rc ? error(rc) : PyTuple_Pack(2, e, st);
    }
    Py_XDECREF(e);
    Py_XDECREF(st);
    return out;
}

/* attention_gnorm_call(q, k, gd, vd, compute layout, flush, sigma ((num, den), or None for 1/sqrt(E)), E, S, Ev, causal,
   mask, mask layout, out layout, rounding, overflow, query slots): a call of the backward's normal answer
   (attention_normal.c), as a capsule */
static PyObject *py_attention_gnorm_call(PyObject *self, PyObject *args) {
    PyObject *qo, *ko, *gdo, *vdo, *compute_layout, *sig, *mask_obj, *mask_layout, *layout;
    Py_ssize_t E, S, Ev, slots;
    int flush, causal, rnd, ovf;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOOpOnnnpOOOiin", &qo, &ko, &gdo, &vdo, &compute_layout, &flush, &sig, &E, &S, &Ev, &causal, &mask_obj, &mask_layout,
                          &layout, &rnd, &ovf, &slots))
        return NULL;
    gnorm_call_box *b = calloc(1, sizeof *b);
    if (!b) return PyErr_NoMemory();
    att_gnorm_call *c = &b->c;
    att_gnorm_call_init(c);
    PyObject *out = NULL;
    partfmt cf = {0}, pf = {0};
    bigint sm;
    big_init(&sm);
    int64_t se = 0;
    int skind = VALUE_FINITE, sneg = 0;
    const unsigned char *mask_data = NULL;
    Py_ssize_t mask_len;
    c->a.q = operand_get(qo);
    c->a.k = operand_get(ko);
    c->gd = operand_get(gdo);
    c->vd = operand_get(vdo);
    if (!c->a.q || !c->a.k || !c->gd || !c->vd || partfmt_load(compute_layout, &cf, c->compute_nan) || partfmt_load(layout, &pf, c->out_nan) ||
        (mask_obj != Py_None && data_load(mask_obj, &mask_data, &mask_len)) || sigma_load(sig, &c->a.num, &c->a.den))
        goto done;
    c->nm.f = cf.f;
    /* the scale held in the compute format (normal_scale), read as the emulation reads a value; its sign for infinite
       scores */
    int rc = femu_set(&c->nm.x, &c->nm.f, ROUND_NEAREST_EVEN, flush);
    if (!rc) rc = normal_scale(&skind, &sneg, &sm, &se, &cf.f, &c->a.num, &c->a.den, E);
    if (!rc && sm.n > 1) rc = EXACTREF_INTERNAL;
    if (rc) {
        error(rc);
        goto done;
    }
    femu_read(&c->nm.sigma, &c->nm.x, skind, sneg, sm.n ? sm.w[0] : 0, se);
    c->a.sign = c->nm.sigma.kind == VALUE_FINITE && !c->nm.sigma.sig ? 0 : c->nm.sigma.neg ? -1 : 1;
    if (mask_obj != Py_None) c->a.mask = mask_data;
    if (mask_load(mask_layout, &c->a)) goto done;
    c->a.S = (size_t)S;
    c->a.Ev = (size_t)Ev;
    c->a.causal = causal;
    c->a.E = E;
    c->out = pf.f;
    c->out_size = part_size(&pf);
    c->rnd = rnd;
    c->ovf = ovf;
    if ((rc = att_gnorm_query_slots(c, (size_t)slots))) {
        error(rc);
        goto done;
    }
    if (!(b->keep = PyTuple_Pack(5, qo, ko, gdo, vdo, mask_obj))) goto done;
    out = PyCapsule_New(b, ATT_GNORM_CALL_CAPSULE, gnorm_call_destroy);
done:
    big_free(&sm);
    if (!out || out == Py_None) {
        att_gnorm_call_free(c);
        Py_XDECREF(b->keep);
        free(b);
    }
    return out;
}

/* attention_gnorm_group(call, kr0, vr0, kblock, vblock, cap): a key and value unit's tile, adding to the two blocks */
static PyObject *py_attention_gnorm_group(PyObject *self, PyObject *args) {
    PyObject *co, *ko, *vo;
    Py_ssize_t kr0, vr0, cap;
    (void)self;
    if (!PyArg_ParseTuple(args, "OnnOOn", &co, &kr0, &vr0, &ko, &vo, &cap)) return NULL;
    gnorm_call_box *cb = PyCapsule_GetPointer(co, ATT_GNORM_CALL_CAPSULE);
    att_gnorm_block *kb = cb ? PyCapsule_GetPointer(ko, ATT_GNORM_BLOCK_CAPSULE) : NULL, *vb = kb ? PyCapsule_GetPointer(vo, ATT_GNORM_BLOCK_CAPSULE) : NULL;
    if (!vb) return NULL;
    gnorm_group_box *b = calloc(1, sizeof *b);
    if (!b) return PyErr_NoMemory();
    att_gnorm_group_init(&b->g);
    int rc = att_gnorm_group_start(&b->g, &cb->c, (size_t)kr0, (size_t)vr0, kb, vb, (size_t)cap);
    PyObject *out = rc ? error(rc) : PyCapsule_New(b, ATT_GNORM_GROUP_CAPSULE, gnorm_group_destroy);
    if (!out) {
        att_gnorm_group_free(&b->g);
        free(b);
        return NULL;
    }
    b->call = Py_NewRef(co);
    b->kb = Py_NewRef(ko);
    b->vb = Py_NewRef(vo);
    return out;
}

static int gnorm_args(PyObject *go, PyObject *co, gnorm_group_box **g, gnorm_call_box **c) {
    *g = PyCapsule_GetPointer(go, ATT_GNORM_GROUP_CAPSULE);
    *c = *g ? PyCapsule_GetPointer(co, ATT_GNORM_CALL_CAPSULE) : NULL;
    return *g && *c ? 0 : -1;
}

/* attention_gnorm_tile(group, rows): the group's next tile, its rows as attention_gtol_tile's */
static PyObject *py_attention_gnorm_tile(PyObject *self, PyObject *args) {
    PyObject *go, *rows_obj;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &go, &rows_obj)) return NULL;
    gnorm_group_box *b = PyCapsule_GetPointer(go, ATT_GNORM_GROUP_CAPSULE);
    const unsigned char *rows;
    Py_ssize_t len;
    if (!b || data_load(rows_obj, &rows, &len)) return NULL;
    size_t T = (size_t)len / (8 * ATT_GR_INFO);
    if (T > b->g.cap) {
        PyErr_SetString(PyExc_ValueError, "attention_gnorm_tile: more rows than the group's tiles hold");
        return NULL;
    }
    memcpy(b->g.rinfo, rows, T * 8 * ATT_GR_INFO);
    b->g.T = T;
    return Py_NewRef(Py_None);
}

static PyObject *gnorm_phase(PyObject *args, int keys) {
    PyObject *go, *co;
    Py_ssize_t x0, x1;
    if (!PyArg_ParseTuple(args, "OOnn", &go, &co, &x0, &x1)) return NULL;
    gnorm_group_box *g;
    gnorm_call_box *c;
    if (gnorm_args(go, co, &g, &c)) return NULL;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = keys ? att_gnorm_tile_keys(&g->g, &c->c, (size_t)x0, (size_t)x1) : att_gnorm_tile_rows(&g->g, &c->c, (size_t)x0, (size_t)x1);
    Py_END_ALLOW_THREADS
    if (rc) return error(rc);
    return Py_NewRef(Py_None);
}

static PyObject *py_attention_gnorm_tile_rows(PyObject *self, PyObject *args) {
    (void)self;
    return gnorm_phase(args, 0);
}

static PyObject *py_attention_gnorm_tile_keys(PyObject *self, PyObject *args) {
    (void)self;
    return gnorm_phase(args, 1);
}

/* attention_gnorm_tile_queries(group, call, r0, r1): rows r0 to r1's grad_query (E per row): (encodings, statuses) */
static PyObject *py_attention_gnorm_tile_queries(PyObject *self, PyObject *args) {
    PyObject *go, *co;
    Py_ssize_t r0, r1;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOnn", &go, &co, &r0, &r1)) return NULL;
    gnorm_group_box *g;
    gnorm_call_box *c;
    if (gnorm_args(go, co, &g, &c)) return NULL;
    size_t n = (size_t)(r1 - r0) * (size_t)c->c.a.E, size = c->c.out_size;
    PyObject *e = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(n * size)), *s = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)n), *out = NULL;
    if (e && s) {
        int rc;
        unsigned char *ed = (unsigned char *)PyBytes_AsString(e), *sd = (unsigned char *)PyBytes_AsString(s);
        Py_BEGIN_ALLOW_THREADS
        rc = att_gnorm_tile_queries(&g->g, &c->c, (size_t)r0, (size_t)r1, ed, sd);
        Py_END_ALLOW_THREADS
        out = rc ? error(rc) : PyTuple_Pack(2, e, s);
    }
    Py_XDECREF(e);
    Py_XDECREF(s);
    return out;
}

/* attention_gnorm_slots(call, which, index, rows): an element whose index order overflowed, from its rows (ATT_GR_INFO
   values, key row, value row each, as native int64), by Lemma O's slots: its encoding */
static PyObject *py_attention_gnorm_slots(PyObject *self, PyObject *args) {
    PyObject *co, *rows_obj;
    int which;
    Py_ssize_t index;
    (void)self;
    if (!PyArg_ParseTuple(args, "OinO", &co, &which, &index, &rows_obj)) return NULL;
    gnorm_call_box *c = PyCapsule_GetPointer(co, ATT_GNORM_CALL_CAPSULE);
    const unsigned char *rows;
    Py_ssize_t len;
    if (!c || data_load(rows_obj, &rows, &len)) return NULL;
    PyObject *e = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)c->c.out_size);
    if (!e) return NULL;
    int rc = att_gnorm_slots((unsigned char *)PyBytes_AsString(e), &c->c, which, (size_t)index, (const int64_t *)rows, (size_t)len / (8 * (ATT_GR_INFO + 2)));
    if (rc) {
        Py_DECREF(e);
        return error(rc);
    }
    return e;
}

static PyObject *py_attention_gtol_call(PyObject *self, PyObject *args) {
    PyObject *gro, *qo, *ko, *gdo, *vdo, *gso, *qso, *kso, *odo, *ldo, *to, *sig, *mask_obj, *mask_layout, *layout, *f64_layout, *refs[3], *radii;
    PyObject *ro[5] = {Py_None, Py_None, Py_None, Py_None, Py_None}, *lrad = Py_None, *mrad = Py_None; /* the radii: Q's, K's, V's, grad's, out's */
    Py_ssize_t E, u, S, Ev;
    long long pmin = 0, dmin = 0;
    int causal, rnd, ovf;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOnnnnpOOOiiOOOOO|LL", &gro, &qo, &ko, &gdo, &vdo, &gso, &qso, &kso, &odo, &ldo, &to, &sig, &E, &u, &S, &Ev,
                          &causal, &mask_obj, &mask_layout, &layout, &rnd, &ovf, &f64_layout, &refs[0], &refs[1], &refs[2], &radii, &pmin, &dmin) ||
        (radii != Py_None && !PyArg_ParseTuple(radii, "OOOOOOO", &ro[0], &ro[1], &ro[2], &ro[3], &ro[4], &lrad, &mrad)))
        return NULL;
    grad_call_box *b = calloc(1, sizeof *b);
    if (!b) return PyErr_NoMemory();
    att_gtol_call *c = &b->c;
    att_gtol_call_init(c);
    PyObject *out = NULL;
    partfmt pf = {0}, f64 = {0};
    c->gr = PyCapsule_GetPointer(gro, ATT_GTOL_CAPSULE);
    c->a.q = operand_get(qo);
    c->a.k = operand_get(ko);
    c->gd = operand_get(gdo);
    c->vd = operand_get(vdo);
    c->gs = operand_get(gso);
    c->qs = operand_get(qso);
    c->ks = operand_get(kso);
    c->od = odo == Py_None ? NULL : operand_get(odo);
    c->ld = ldo == Py_None ? NULL : operand_get(ldo);
    c->a.tables = to == Py_None ? NULL : PyCapsule_GetPointer(to, ATT_TABLES_CAPSULE);
    /* inputs that are earlier results: their radii as operands, the given logsumexp's and the float mask's as float64 */
    const operand *rops[5] = {NULL, NULL, NULL, NULL, NULL};
    for (int z = 0; z < 5; z++)
        if (ro[z] != Py_None && !(rops[z] = operand_get(ro[z]))) goto done;
    c->a.rq = rops[0];
    c->a.rk = rops[1];
    c->rv = rops[2];
    c->rg = rops[3];
    c->ro = rops[4];
    const unsigned char *mask_data = NULL;
    Py_ssize_t mask_len, len;
    if ((lrad != Py_None && data_load(lrad, &c->lrad, &len)) || (mrad != Py_None && data_load(mrad, &c->a.mrad, &len))) goto done;
    c->a.uncertain = c->a.rq || c->a.rk || c->a.mrad;
    if (!c->gr || !c->a.q || !c->a.k || !c->gd || !c->vd || !c->gs || !c->qs || !c->ks || (odo != Py_None && !c->od) || (ldo != Py_None && !c->ld) ||
        (to != Py_None && !c->a.tables) || sigma_load(sig, &c->a.num, &c->a.den) ||
        (mask_obj != Py_None && data_load(mask_obj, &mask_data, &mask_len)) || partfmt_load(layout, &pf, c->out_nan) || partfmt_load(f64_layout, &f64, c->f64_nan))
        goto done;
    for (int z = 0; z < 3; z++)
        if (data_load(refs[z], &c->ref[z], &len)) goto done;
    if (mask_obj != Py_None) c->a.mask = mask_data;
    if (mask_load(mask_layout, &c->a)) goto done;
    c->a.mtables = c->a.tables ? c->a.tables + 1 : NULL;
    c->a.S = (size_t)S;
    c->a.Ev = (size_t)Ev;
    c->a.causal = causal;
    c->a.sign = !c->a.den.n ? 1 : !c->a.num.n ? 0 : c->a.num.neg ? -1 : 1;
    c->a.E = E;
    c->a.u = u;
    c->a.p = c->gr->w;
    c->out = pf.f;
    c->f64 = f64.f;
    c->out_size = part_size(&pf);
    c->rnd = rnd;
    c->ovf = ovf;
    c->pmin = pmin;
    c->dmin = dmin;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = att_gtol_call_prepare(c);
    Py_END_ALLOW_THREADS
    if (rc) {
        error(rc);
        goto done;
    }
    if (!(b->keep = PyTuple_Pack(16, gro, qo, ko, gdo, vdo, gso, qso, kso, odo, ldo, to, mask_obj, refs[0], refs[1], refs[2], radii))) goto done;
    out = PyCapsule_New(b, ATT_GTOL_CALL_CAPSULE, grad_call_destroy);
done:
    if (!out) {
        att_gtol_call_free(c);
        Py_XDECREF(b->keep);
        free(b);
    }
    return out;
}

#define ATT_GTOL_BLOCK_CAPSULE "exactref._c.att_gtol_block"

static void grad_block_destroy(PyObject *capsule) {
    att_gtol_block *b = PyCapsule_GetPointer(capsule, ATT_GTOL_BLOCK_CAPSULE);
    if (!b) return;
    att_gtol_block_free(b);
    free(b);
}

/* attention_gtol_block(call, key, r0): a key block's grad_key sums (key, K rows r0..) or a value block's grad_value sums
   (V rows r0..), as a capsule */
static PyObject *py_attention_gtol_block(PyObject *self, PyObject *args) {
    PyObject *co;
    int key;
    Py_ssize_t r0;
    (void)self;
    if (!PyArg_ParseTuple(args, "Opn", &co, &key, &r0)) return NULL;
    grad_call_box *cb = PyCapsule_GetPointer(co, ATT_GTOL_CALL_CAPSULE);
    if (!cb) return NULL;
    att_gtol_block *b = malloc(sizeof *b);
    if (!b) return PyErr_NoMemory();
    int rc = att_gtol_block_start(b, &cb->c, key, (size_t)r0);
    PyObject *out = rc ? error(rc) : PyCapsule_New(b, ATT_GTOL_BLOCK_CAPSULE, grad_block_destroy);
    if (!out) {
        att_gtol_block_free(b);
        free(b);
    }
    return out;
}

/* attention_gtol_group(call, kr0, vr0, kblock, vblock, cap): a key and value unit's tile, adding to the two blocks */
static PyObject *py_attention_gtol_group(PyObject *self, PyObject *args) {
    PyObject *co, *ko, *vo;
    Py_ssize_t kr0, vr0, cap;
    (void)self;
    if (!PyArg_ParseTuple(args, "OnnOOn", &co, &kr0, &vr0, &ko, &vo, &cap)) return NULL;
    grad_call_box *cb = PyCapsule_GetPointer(co, ATT_GTOL_CALL_CAPSULE);
    att_gtol_block *kb = cb ? PyCapsule_GetPointer(ko, ATT_GTOL_BLOCK_CAPSULE) : NULL, *vb = kb ? PyCapsule_GetPointer(vo, ATT_GTOL_BLOCK_CAPSULE) : NULL;
    if (!vb) return NULL;
    grad_group_box *b = calloc(1, sizeof *b);
    if (!b) return PyErr_NoMemory();
    att_gtol_group_init(&b->g);
    int rc = att_gtol_group_start(&b->g, &cb->c, (size_t)kr0, (size_t)vr0, kb, vb, (size_t)cap);
    PyObject *out = rc ? error(rc) : PyCapsule_New(b, ATT_GTOL_GROUP_CAPSULE, grad_group_destroy);
    if (!out) {
        att_gtol_group_free(&b->g);
        free(b);
        return NULL;
    }
    b->call = Py_NewRef(co);
    b->kb = Py_NewRef(ko);
    b->vb = Py_NewRef(vo);
    return out;
}

static int grad_args(PyObject *go, PyObject *co, grad_group_box **g, grad_call_box **c) {
    *g = PyCapsule_GetPointer(go, ATT_GTOL_GROUP_CAPSULE);
    *c = *g ? PyCapsule_GetPointer(co, ATT_GTOL_CALL_CAPSULE) : NULL;
    return *g && *c ? 0 : -1;
}

/* attention_gtol_tile(group, rows): the group's next tile, its rows as (query row, query index, mask offset, mask step,
   output-gradient row, broadcast slot or -1, copy, copies) in native int64 */
static PyObject *py_attention_gtol_tile(PyObject *self, PyObject *args) {
    PyObject *go, *rows_obj;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &go, &rows_obj)) return NULL;
    grad_group_box *b = PyCapsule_GetPointer(go, ATT_GTOL_GROUP_CAPSULE);
    const unsigned char *rows;
    Py_ssize_t len;
    if (!b || data_load(rows_obj, &rows, &len)) return NULL;
    size_t T = (size_t)len / (8 * ATT_GR_INFO);
    if (T > b->g.cap) {
        PyErr_SetString(PyExc_ValueError, "attention_gtol_tile: more rows than the group's tiles hold");
        return NULL;
    }
    memcpy(b->g.rinfo, rows, T * 8 * ATT_GR_INFO);
    b->g.T = T;
    return Py_NewRef(Py_None);
}

static PyObject *py_attention_gtol_tile_rows(PyObject *self, PyObject *args) {
    PyObject *go, *co;
    Py_ssize_t r0, r1;
    (void)self;
    grad_group_box *g;
    grad_call_box *c;
    if (!PyArg_ParseTuple(args, "OOnn", &go, &co, &r0, &r1) || grad_args(go, co, &g, &c)) return NULL;
    int rc;
    size_t closed;
    Py_BEGIN_ALLOW_THREADS
    rc = att_gtol_tile_rows(&closed, &g->g, &c->c, (size_t)r0, (size_t)r1);
    Py_END_ALLOW_THREADS
    if (rc) return error(rc);
    return PyLong_FromSize_t(closed);
}

/* attention_gtol_tile_need(group): None where every P' and dS' end of the tile's rows fit the bits the call's sums are
   sized for, else (P' bits, dS' bits), the most its rows need (0 where they fit), for the call made again */
static PyObject *py_attention_gtol_tile_need(PyObject *self, PyObject *args) {
    PyObject *go;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &go)) return NULL;
    grad_group_box *b = PyCapsule_GetPointer(go, ATT_GTOL_GROUP_CAPSULE);
    if (!b) return NULL;
    long long need[2] = {0, 0};
    for (size_t r = 0; r < b->g.T; r++)
        for (int z = 0; z < 2; z++)
            if (b->g.need[2 * r + z] > need[z]) need[z] = b->g.need[2 * r + z];
    if (!need[0] && !need[1]) return Py_NewRef(Py_None);
    return Py_BuildValue("(LL)", need[0], need[1]);
}

static PyObject *py_attention_gtol_tile_keys(PyObject *self, PyObject *args) {
    PyObject *go, *co;
    Py_ssize_t j0, j1;
    (void)self;
    grad_group_box *g;
    grad_call_box *c;
    if (!PyArg_ParseTuple(args, "OOnn", &go, &co, &j0, &j1) || grad_args(go, co, &g, &c)) return NULL;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = att_gtol_tile_keys(&g->g, &c->c, (size_t)j0, (size_t)j1);
    Py_END_ALLOW_THREADS
    if (rc) return error(rc);
    return Py_NewRef(Py_None);
}

/* the four parts of count elements' bounds (tolerance, low, high, flags), as new bytes */
static int parts_new(PyObject **res, unsigned char **o, size_t count, size_t size) {
    Py_ssize_t sizes[4] = {(Py_ssize_t)(count * 8), (Py_ssize_t)(count * size), (Py_ssize_t)(count * size), (Py_ssize_t)count};
    for (int z = 0; z < 4; z++) {
        if (!(res[z] = PyBytes_FromStringAndSize(NULL, sizes[z]))) return -1;
        o[z] = (unsigned char *)PyBytes_AsString(res[z]);
        memset(o[z], 0, (size_t)sizes[z]);
    }
    return 0;
}

static PyObject *py_attention_gtol_tile_queries(PyObject *self, PyObject *args) {
    PyObject *go, *co, *res[4] = {NULL}, *out = NULL;
    Py_ssize_t r0, r1;
    (void)self;
    grad_group_box *g;
    grad_call_box *c;
    if (!PyArg_ParseTuple(args, "OOnn", &go, &co, &r0, &r1) || grad_args(go, co, &g, &c)) return NULL;
    unsigned char *o[4];
    if (!parts_new(res, o, (size_t)(r1 - r0) * (size_t)c->c.a.E, c->c.out_size)) {
        int rc;
        Py_BEGIN_ALLOW_THREADS
        rc = att_gtol_tile_queries(&g->g, &c->c, (size_t)r0, (size_t)r1, o[0], o[1], o[2], o[3]);
        Py_END_ALLOW_THREADS
        out = rc ? error(rc) : PyTuple_Pack(4, res[0], res[1], res[2], res[3]);
    }
    for (int z = 0; z < 4; z++) Py_XDECREF(res[z]);
    return out;
}

static PyObject *py_attention_gtol_query_slots(PyObject *self, PyObject *args) {
    PyObject *co;
    Py_ssize_t count;
    (void)self;
    if (!PyArg_ParseTuple(args, "On", &co, &count)) return NULL;
    grad_call_box *c = PyCapsule_GetPointer(co, ATT_GTOL_CALL_CAPSULE);
    if (!c) return NULL;
    int rc = att_gtol_query_slots(&c->c, (size_t)count);
    if (rc) return error(rc);
    return Py_NewRef(Py_None);
}

static PyObject *py_attention_gtol_tile_done(PyObject *self, PyObject *args) {
    PyObject *go, *co;
    (void)self;
    grad_group_box *g;
    grad_call_box *c;
    if (!PyArg_ParseTuple(args, "OO", &go, &co) || grad_args(go, co, &g, &c)) return NULL;
    int rc = att_gtol_tile_done(&g->g, &c->c);
    if (rc) return error(rc);
    return Py_NewRef(Py_None);
}

static PyObject *py_attention_gtol_group_carry(PyObject *self, PyObject *args) {
    PyObject *go, *co;
    Py_ssize_t j0, j1;
    (void)self;
    grad_group_box *g;
    grad_call_box *c;
    if (!PyArg_ParseTuple(args, "OOnn", &go, &co, &j0, &j1) || grad_args(go, co, &g, &c)) return NULL;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = att_gtol_group_carry(&g->g, &c->c, (size_t)j0, (size_t)j1);
    Py_END_ALLOW_THREADS
    if (rc) return error(rc);
    return Py_NewRef(Py_None);
}

static PyObject *py_attention_gtol_group_carried(PyObject *self, PyObject *args) {
    PyObject *go;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &go)) return NULL;
    grad_group_box *b = PyCapsule_GetPointer(go, ATT_GTOL_GROUP_CAPSULE);
    if (!b) return NULL;
    int rc = att_gtol_group_carried(&b->g);
    if (rc) return error(rc);
    return Py_NewRef(Py_None);
}

static PyObject *py_attention_gtol_group_fold(PyObject *self, PyObject *args) {
    PyObject *go, *co;
    Py_ssize_t j0, j1;
    (void)self;
    grad_group_box *g;
    grad_call_box *c;
    if (!PyArg_ParseTuple(args, "OOnn", &go, &co, &j0, &j1) || grad_args(go, co, &g, &c)) return NULL;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = att_gtol_group_fold(&g->g, &c->c, (size_t)j0, (size_t)j1);
    Py_END_ALLOW_THREADS
    if (rc) return error(rc);
    return Py_NewRef(Py_None);
}

static PyObject *py_attention_gtol_group_folded(PyObject *self, PyObject *args) {
    PyObject *go;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &go)) return NULL;
    grad_group_box *b = PyCapsule_GetPointer(go, ATT_GTOL_GROUP_CAPSULE);
    if (!b) return NULL;
    att_gtol_group_folded(&b->g);
    return Py_NewRef(Py_None);
}

/* attention_gtol_block_finish(block, call, key, j0, j1): keys j0 to j1's elements after the block's last unit: (tolerances,
   lows, highs, flags) */
static PyObject *py_attention_gtol_block_finish(PyObject *self, PyObject *args) {
    PyObject *bo, *co, *res[4] = {NULL}, *out = NULL;
    int key;
    Py_ssize_t j0, j1;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOpnn", &bo, &co, &key, &j0, &j1)) return NULL;
    att_gtol_block *b = PyCapsule_GetPointer(bo, ATT_GTOL_BLOCK_CAPSULE);
    grad_call_box *c = b ? PyCapsule_GetPointer(co, ATT_GTOL_CALL_CAPSULE) : NULL;
    if (!c) return NULL;
    unsigned char *o[4];
    if (!parts_new(res, o, (size_t)(j1 - j0) * b->cols, c->c.out_size)) {
        int rc;
        Py_BEGIN_ALLOW_THREADS
        rc = att_gtol_block_finish(b, &c->c, key, (size_t)j0, (size_t)j1, o[0], o[1], o[2], o[3]);
        Py_END_ALLOW_THREADS
        out = rc ? error(rc) : Py_BuildValue("(OOOO)", res[0], res[1], res[2], res[3]);
    }
    for (int z = 0; z < 4; z++) Py_XDECREF(res[z]);
    return out;
}

static PyObject *py_attention_program_build(PyObject *self, PyObject *args) {
    PyObject *so, *f64_layout;
    int sign;
    (void)self;
    if (!PyArg_ParseTuple(args, "OiO", &so, &sign, &f64_layout)) return NULL;
    const att_settings *s = PyCapsule_GetPointer(so, ATT_SETTINGS_CAPSULE);
    if (!s) return NULL;
    att_prog *g = malloc(sizeof *g);
    if (!g) return PyErr_NoMemory();
    att_prog_init(g);
    int applicable = 0, bad = -1, rc = fmt_load(f64_layout, &g->f64) ? -1 : 0;
    if (!rc) {
        Py_BEGIN_ALLOW_THREADS
        rc = att_prog_build(g, &applicable, &bad, s, sign);
        if (!rc && applicable) rc = att_prog_prepare(g, &s->a[ATT_SC_SCORES].f, s->E);
        Py_END_ALLOW_THREADS
    }
    PyObject *out = NULL;
    if (!rc && applicable && (out = PyCapsule_New(g, ATT_PROG_CAPSULE, att_prog_destroy))) return out;
    att_prog_free(g);
    free(g);
    if (rc > 0) return settings_error(rc, bad);
    if (rc || PyErr_Occurred()) return NULL; /* PyCapsule_New's failure set its exception */
    return Py_NewRef(Py_None);
}

static PyObject *py_attention_program(PyObject *self, PyObject *args) {
    PyObject *qo, *ko, *vo, *go, *rows_obj, *mask_obj, *mask_layout, *layout, *lse_layout, *refs, *lse_refs, *rqo, *rko, *vvo, *vro, *vunbo, *mrado, *to,
        *sig;
    Py_ssize_t E, kr0, vr0, S, Ev;
    int causal, rnd, ovf, lrnd, lovf;
    long long ve, u;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOnOnnnnpOOOiiOiiOOOOOOLOOOLO", &qo, &ko, &vo, &go, &E, &rows_obj, &kr0, &vr0, &S, &Ev, &causal, &mask_obj,
                          &mask_layout, &layout, &rnd, &ovf, &lse_layout, &lrnd, &lovf, &refs, &lse_refs, &rqo, &rko, &vvo, &vro, &ve, &vunbo, &mrado, &to,
                          &u, &sig))
        return NULL;
    att_call a;
    memset(&a, 0, sizeof a);
    a.q = operand_get(qo);
    a.k = operand_get(ko);
    a.v = operand_get(vo);
    const att_prog *g = PyCapsule_GetPointer(go, ATT_PROG_CAPSULE);
    if (!a.q || !a.k || !a.v || !g) return NULL;
    /* inputs that are earlier results: their radii */
    if ((rqo != Py_None && !(a.rq = operand_get(rqo))) || (rko != Py_None && !(a.rk = operand_get(rko)))) return NULL;
    Py_ssize_t radii_len;
    if ((vvo != Py_None && (data_load(vvo, &a.vv, &radii_len) || data_load(vro, &a.vr, &radii_len) || data_load(vunbo, &a.vunb, &radii_len))) ||
        (mrado != Py_None && data_load(mrado, &a.mrad, &radii_len)))
        return NULL;
    a.ve = ve;
    a.uncertain = rqo != Py_None || rko != Py_None || vvo != Py_None || mrado != Py_None;
    partfmt pf = {0}, lf = {0};
    bigint nan[2], lnan[2];
    for (int z = 0; z < 2; z++) {
        big_init(&nan[z]);
        big_init(&lnan[z]);
    }
    big_init(&a.num);
    big_init(&a.den);
    const unsigned char *rows, *rf, *lrf;
    Py_ssize_t len, n1, n2;
    PyObject *out = NULL, *res[9] = {NULL};
    const unsigned char *mask_data = NULL;
    Py_ssize_t mask_len;
    if (partfmt_load(layout, &pf, nan) || partfmt_load(lse_layout, &lf, lnan) || data_load(rows_obj, &rows, &len) || data_load(refs, &rf, &n1) ||
        data_load(lse_refs, &lrf, &n2) || (mask_obj != Py_None && data_load(mask_obj, &mask_data, &mask_len)))
        goto done;
    if (mask_obj != Py_None) a.mask = mask_data;
    if (mask_load(mask_layout, &a)) goto done;
    a.S = (size_t)S;
    a.Ev = (size_t)Ev;
    a.causal = causal;
    a.sign = g->sign;
    a.E = E;
    a.u = u;
    a.p = g->w;
    /* Lemma D's tables for the program's weights (None: Lemma X per weight) */
    a.tables = to == Py_None ? NULL : PyCapsule_GetPointer(to, ATT_TABLES_CAPSULE);
    if ((to != Py_None && !a.tables) || sigma_load(sig, &a.num, &a.den)) goto done;
    a.out = pf.f;
    a.lse = lf.f;
    a.out_size = part_size(&pf);
    a.lse_size = part_size(&lf);
    a.rnd = rnd;
    a.ovf = ovf;
    a.lrnd = lrnd;
    a.lovf = lovf;
    size_t count = (size_t)len / 32, cells = count * a.Ev;
    Py_ssize_t sizes[9] = {(Py_ssize_t)count, (Py_ssize_t)(cells * 8), (Py_ssize_t)(cells * a.out_size), (Py_ssize_t)(cells * a.out_size),
                           (Py_ssize_t)cells, (Py_ssize_t)(count * 8), (Py_ssize_t)(count * a.lse_size), (Py_ssize_t)(count * a.lse_size),
                           (Py_ssize_t)count};
    for (int z = 0; z < 9; z++)
        if (!(res[z] = PyBytes_FromStringAndSize(NULL, sizes[z]))) goto done;
    unsigned char *o[9];
    for (int z = 0; z < 9; z++) o[z] = (unsigned char *)PyBytes_AsString(res[z]);
    for (int z = 1; z < 9; z++) memset(o[z], 0, (size_t)sizes[z]);
    int rc = 0;
    Py_BEGIN_ALLOW_THREADS
    att_row w;
    att_row_init(&w);
    for (size_t r = 0; !rc && r < count; r++) {
        int64_t qr, i, moff, mstep;
        memcpy(&qr, rows + 32 * r, 8);
        memcpy(&i, rows + 32 * r + 8, 8);
        memcpy(&moff, rows + 32 * r + 16, 8);
        memcpy(&mstep, rows + 32 * r + 24, 8);
        rc = att_program_row(&w, &a, g, (size_t)qr, (size_t)i, (size_t)moff, (size_t)mstep, (size_t)kr0, (size_t)vr0, rf + r * a.Ev * a.out_size,
                             lrf + r * a.lse_size, o[1] + r * a.Ev * 8, o[2] + r * a.Ev * a.out_size, o[3] + r * a.Ev * a.out_size, o[4] + r * a.Ev,
                             o[5] + r * 8, o[6] + r * a.lse_size, o[7] + r * a.lse_size, o[8] + r, o[0] + r);
    }
    att_row_free(&w);
    Py_END_ALLOW_THREADS
    (void)n1;
    (void)n2;
    (void)mask_len;
    if (rc)
        error(rc);
    else
        out = Py_BuildValue("(OOOOOOOOO)", res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8]);
done:
    for (int z = 0; z < 9; z++) Py_XDECREF(res[z]);
    for (int z = 0; z < 2; z++) {
        big_free(&nan[z]);
        big_free(&lnan[z]);
    }
    big_free(&a.num);
    big_free(&a.den);
    return out;
}

/* The attention forward's normal answer (attention_normal.c) */
static PyObject *py_attention_normal(PyObject *self, PyObject *args) {
    PyObject *qo, *ko, *vo, *compute_layout, *sig, *rows_obj, *mask_obj, *mask_layout, *layout, *lse_layout;
    Py_ssize_t E, kr0, vr0, S, Ev;
    int flush, causal, rnd, ovf, lrnd, lovf;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOpOnOnnnnpOOOiiOii", &qo, &ko, &vo, &compute_layout, &flush, &sig, &E, &rows_obj, &kr0, &vr0, &S, &Ev, &causal,
                          &mask_obj, &mask_layout, &layout, &rnd, &ovf, &lse_layout, &lrnd, &lovf))
        return NULL;
    att_call a;
    memset(&a, 0, sizeof a);
    a.q = operand_get(qo);
    a.k = operand_get(ko);
    a.v = operand_get(vo);
    if (!a.q || !a.k || !a.v) return NULL;
    att_normal nm;
    partfmt cf = {0}, pf = {0}, lf = {0};
    bigint cnan[2], nan[2], lnan[2], sm;
    for (int z = 0; z < 2; z++) {
        big_init(&cnan[z]);
        big_init(&nan[z]);
        big_init(&lnan[z]);
    }
    big_init(&sm);
    big_init(&a.num);
    big_init(&a.den);
    PyObject *out = NULL, *res[2] = {NULL, NULL};
    int64_t se = 0;
    int skind = VALUE_FINITE, sneg = 0;
    const unsigned char *rows, *mask_data = NULL;
    Py_ssize_t len, mask_len;
    if (partfmt_load(compute_layout, &cf, cnan) || partfmt_load(layout, &pf, nan) || partfmt_load(lse_layout, &lf, lnan) ||
        data_load(rows_obj, &rows, &len) || (mask_obj != Py_None && data_load(mask_obj, &mask_data, &mask_len)) ||
        sigma_load(sig, &a.num, &a.den))
        goto done;
    nm.f = cf.f;
    {
        /* the scale held in the compute format (normal_scale), read as the emulation reads a value; its sign for
           infinite scores */
        int rc = femu_set(&nm.x, &nm.f, ROUND_NEAREST_EVEN, flush);
        if (!rc) rc = normal_scale(&skind, &sneg, &sm, &se, &cf.f, &a.num, &a.den, E);
        if (!rc && sm.n > 1) rc = EXACTREF_INTERNAL;
        if (rc) {
            error(rc);
            goto done;
        }
    }
    femu_read(&nm.sigma, &nm.x, skind, sneg, sm.n ? sm.w[0] : 0, se);
    a.sign = nm.sigma.kind == VALUE_FINITE && !nm.sigma.sig ? 0 : nm.sigma.neg ? -1 : 1;
    if (mask_obj != Py_None) a.mask = mask_data;
    if (mask_load(mask_layout, &a)) goto done;
    a.S = (size_t)S;
    a.Ev = (size_t)Ev;
    a.causal = causal;
    a.E = E;
    a.out = pf.f;
    a.lse = lf.f;
    a.out_size = part_size(&pf);
    a.lse_size = part_size(&lf);
    a.rnd = rnd;
    a.ovf = ovf;
    a.lrnd = lrnd;
    a.lovf = lovf;
    size_t count = (size_t)len / 32;
    if (!(res[0] = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(count * a.Ev * a.out_size))) ||
        !(res[1] = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(count * a.lse_size))))
        goto done;
    unsigned char *enc = (unsigned char *)PyBytes_AsString(res[0]), *lenc = (unsigned char *)PyBytes_AsString(res[1]);
    int rc = 0;
    Py_BEGIN_ALLOW_THREADS
    for (size_t r = 0; !rc && r < count; r++) {
        int64_t qr, i, moff, mstep;
        memcpy(&qr, rows + 32 * r, 8);
        memcpy(&i, rows + 32 * r + 8, 8);
        memcpy(&moff, rows + 32 * r + 16, 8);
        memcpy(&mstep, rows + 32 * r + 24, 8);
        rc = att_normal_row(&a, &nm, (size_t)qr, (size_t)i, (size_t)moff, (size_t)mstep, (size_t)kr0, (size_t)vr0, enc + r * a.Ev * a.out_size,
                            lenc + r * a.lse_size);
    }
    Py_END_ALLOW_THREADS
    (void)mask_len;
    if (rc)
        error(rc);
    else
        out = Py_BuildValue("(OO)", res[0], res[1]);
done:
    Py_XDECREF(res[0]);
    Py_XDECREF(res[1]);
    for (int z = 0; z < 2; z++) {
        big_free(&cnan[z]);
        big_free(&nan[z]);
        big_free(&lnan[z]);
    }
    big_free(&sm);
    big_free(&a.num);
    big_free(&a.den);
    return out;
}

static PyMethodDef methods[] = {
    {"physical_cores", py_physical_cores, METH_NOARGS, "The default number of worker threads."},
    {"attention_tolerance", py_attention_tolerance, METH_VARARGS,
     "attention_tolerance(q, k, v, vt, tables, config, sigma, E, u, rows, kr0, vr0, S, Ev, causal, mask, mask_layout, layout, rnd, ovf, lse_layout, lrnd, lovf, "
     "refs, lse_refs, entry_flags, entry_mids, lse_flags, lse_mids, rq, rk, vv, vr, ve, vunb, mrad): for rows as attention_reference's (the radii as "
     "attention_program takes them), per row its status (0 bounded, 1 left to the caller), and the outputs' and "
     "logsumexp's final bounds: (status, tol, low, high, flags, lse_tol, lse_low, lse_high, lse_flags)."},
    {"attention_settings", py_attention_settings, METH_VARARGS,
     "attention_settings(rounds, accumulations, flush, emin, w, S, E, scale, methods, backward): the forward's or the backward's "
     "settings, its boundaries' and accumulations' (in attention_tol.c's ATT_SR_ and ATT_SC_ order), as a capsule with the factor the "
     "scores are multiplied by and the scale."},
    {"attention_gtol_build", py_attention_gtol_build, METH_VARARGS,
     "attention_gtol_build(settings, normal, closed, sign, f64 layout, Ev, nq, nkv, heads): the backward's constants (attention_grad_tol.c), "
     "as a capsule; without closed every row takes the step-by-step program."},
    {"attention_gtol_call", py_attention_gtol_call, METH_VARARGS,
     "attention_gtol_call(grad, q, k, gd, vd, gs, qs, ks, out, lse, tables, scale, E, u, S, Ev, causal, mask, mask layout, out layout, rounding, "
     "overflow, f64 layout, grad_value refs, grad_key refs, grad_query refs, radii[, pmin, dmin]): a call of the backward's tolerance "
     "(attention_grad_tol.c), as a capsule; out and lse the forward's given output and logsumexp, or None; radii None, or for inputs that are "
     "earlier results (Q's, K's, V's, grad's and out's radii as operands, lse's and the float mask's as float64 bytes), each None where exact; "
     "pmin and dmin the bits the P' and dS' ends are sized for (0 by default; raised to what attention_gtol_tile_need reports on a retry)."},
    {"attention_gnorm_call", py_attention_gnorm_call, METH_VARARGS,
     "attention_gnorm_call(q, k, gd, vd, compute layout, flush, sigma, E, S, Ev, causal, mask, mask layout, out layout, rounding, overflow, query "
     "slots): a call of the backward's normal answer (attention_normal.c), as a capsule."},
    {"attention_gnorm_group", py_attention_gnorm_group, METH_VARARGS,
     "attention_gnorm_group(call, kr0, vr0, kblock, vblock, cap): a key and value unit's tile, adding to the two blocks, as a capsule."},
    {"attention_gnorm_tile", py_attention_gnorm_tile, METH_VARARGS, "attention_gnorm_tile(group, rows): the group's next tile's rows."},
    {"attention_gnorm_tile_rows", py_attention_gnorm_tile_rows, METH_VARARGS, "attention_gnorm_tile_rows(group, call, r0, r1): the tile's rows r0 to r1."},
    {"attention_gnorm_tile_keys", py_attention_gnorm_tile_keys, METH_VARARGS, "attention_gnorm_tile_keys(group, call, j0, j1): the tile's terms of keys j0 to j1."},
    {"attention_gnorm_tile_queries", py_attention_gnorm_tile_queries, METH_VARARGS,
     "attention_gnorm_tile_queries(group, call, r0, r1): rows r0 to r1's grad_query: (encodings, statuses)."},
    {"attention_gnorm_block", py_attention_gnorm_block, METH_VARARGS,
     "attention_gnorm_block(call, cols): a key block's grad_key sums or a value block's grad_value sums, as a capsule."},
    {"attention_gnorm_block_finish", py_attention_gnorm_block_finish, METH_VARARGS,
     "attention_gnorm_block_finish(block, call, scaled, j0, j1): keys j0 to j1's elements after the block's last unit: (encodings, statuses)."},
    {"attention_gnorm_slots", py_attention_gnorm_slots, METH_VARARGS, "attention_gnorm_slots(call, which, index, rows): an overflowed element by Lemma O's slots."},
    {"attention_gref_refine", py_attention_gref_refine, METH_VARARGS,
     "attention_gref_refine(call, rows, elems, rowlist, p0, cap): the backward's refinement of its undecided elements as a run, in waves of "
     "tiles (attention_gref_refine_next, attention_gref_refine_tile, attention_gref_refine_result)."},
    {"attention_gref_refine_next", py_attention_gref_refine_next, METH_VARARGS, "attention_gref_refine_next(run): the tiles of the next wave, 0 when done."},
    {"attention_gref_refine_tile", py_attention_gref_refine_tile, METH_VARARGS, "attention_gref_refine_tile(run, t): runs tile t of the current wave."},
    {"attention_gref_refine_waves", py_attention_gref_refine_waves, METH_VARARGS,
     "attention_gref_refine_waves(run): (its waves so far, those that ran split), for tests."},
    {"attention_gref_refine_result", py_attention_gref_refine_result, METH_VARARGS,
     "attention_gref_refine_result(run): per element its encoding, entry flag, entry midpoint and status (1: undecided at the cap)."},
    {"attention_gref_call", py_attention_gref_call, METH_VARARGS,
     "attention_gref_call(q, k, gd, vd, gs, qs, ks, tables, scale, E, u, S, Ev, causal, mask, mask layout, out layout, rounding, overflow, p, nq, "
     "nkv, query slots): a call of the backward's reference (attention_gref.c), as a capsule."},
    {"attention_gref_group", py_attention_gref_group, METH_VARARGS,
     "attention_gref_group(call, kr0, vr0, kblock, vblock, cap): a key and value unit's tile, adding to the two blocks, as a capsule."},
    {"attention_gref_tile", py_attention_gref_tile, METH_VARARGS, "attention_gref_tile(group, rows): the group's next tile's rows."},
    {"attention_gref_tile_rows", py_attention_gref_tile_rows, METH_VARARGS, "attention_gref_tile_rows(group, call, r0, r1): the tile's rows r0 to r1."},
    {"attention_gref_tile_prep", py_attention_gref_tile_prep, METH_VARARGS, "attention_gref_tile_prep(group, call): the tile's common denominators."},
    {"attention_gref_tile_keys", py_attention_gref_tile_keys, METH_VARARGS, "attention_gref_tile_keys(group, call, j0, j1): the tile's terms of keys j0 to j1."},
    {"attention_gref_tile_queries", py_attention_gref_tile_queries, METH_VARARGS,
     "attention_gref_tile_queries(group, call, r0, r1): rows r0 to r1's grad_query: (encodings, entry flags, entry midpoints, statuses)."},
    {"attention_gref_tile_done", py_attention_gref_tile_done, METH_VARARGS, "attention_gref_tile_done(group): the tile's end."},
    {"attention_gref_block", py_attention_gref_block, METH_VARARGS,
     "attention_gref_block(call, key): a key block's grad_key elements or a value block's grad_value elements, as a capsule."},
    {"attention_gref_block_finish", py_attention_gref_block_finish, METH_VARARGS,
     "attention_gref_block_finish(block, call, key, j0, j1): keys j0 to j1's elements after the block's last unit: (encodings, entry flags, entry "
     "midpoints, statuses)."},
    {"attention_gtol_group", py_attention_gtol_group, METH_VARARGS,
     "attention_gtol_group(call, kr0, vr0, kblock, vblock, cap): a key and value unit's tile, adding to the two blocks, as a capsule."},
    {"attention_gtol_tile", py_attention_gtol_tile, METH_VARARGS, "attention_gtol_tile(group, rows): the group's next tile's rows."},
    {"attention_gtol_tile_rows", py_attention_gtol_tile_rows, METH_VARARGS,
     "attention_gtol_tile_rows(group, call, r0, r1): the tile's rows' P' and dS'; how many of them took P' from the closed form."},
    {"attention_gtol_tile_keys", py_attention_gtol_tile_keys, METH_VARARGS,
     "attention_gtol_tile_keys(group, call, j0, j1): the tile's terms into keys j0 to j1's grad_key and grad_value sums."},
    {"attention_gtol_tile_queries", py_attention_gtol_tile_queries, METH_VARARGS,
     "attention_gtol_tile_queries(group, call, r0, r1): grad_query of the tile's rows r0 to r1, finished: (tolerance, low, high, flags) bytes."},
    {"attention_gtol_tile_done", py_attention_gtol_tile_done, METH_VARARGS, "attention_gtol_tile_done(group, call): the tile's rows join the group."},
    {"attention_gtol_tile_need", py_attention_gtol_tile_need, METH_VARARGS,
     "attention_gtol_tile_need(group): None, or the bits the tile's P' and dS' ends need past those the call's sums are sized for."},
    {"attention_gtol_query_slots", py_attention_gtol_query_slots, METH_VARARGS,
     "attention_gtol_query_slots(call, count): slots for count query rows broadcast over several key and value blocks."},
    {"attention_gtol_group_carry", py_attention_gtol_group_carry, METH_VARARGS,
     "attention_gtol_group_carry(group, call, j0, j1): backward_blocks: the block's sums of keys j0 to j1 added to their carried values."},
    {"attention_gtol_group_carried", py_attention_gtol_group_carried, METH_VARARGS, "attention_gtol_group_carried(group): a block carried."},
    {"attention_gtol_group_fold", py_attention_gtol_group_fold, METH_VARARGS,
     "attention_gtol_group_fold(group, call, j0, j1): grouped_heads: the head's sums of keys j0 to j1 rounded and added to the heads' sums."},
    {"attention_gtol_group_folded", py_attention_gtol_group_folded, METH_VARARGS, "attention_gtol_group_folded(group): one more head folded."},
    {"attention_gtol_block_finish", py_attention_gtol_block_finish, METH_VARARGS,
     "attention_gtol_block_finish(block, call, key, j0, j1): keys j0 to j1's elements after the block's last unit: (tolerances, lows, highs, flags)."},
    {"attention_gtol_block", py_attention_gtol_block, METH_VARARGS,
     "attention_gtol_block(call, key, r0): a key block's grad_key sums or a value block's grad_value sums, as a capsule."},
    {"attention_tolerance_build", py_attention_tolerance_build, METH_VARARGS,
     "attention_tolerance_build(settings, normal, f64_layout): the closed form's configuration from the settings, or None where it does not cover them."},
    {"attention_program_build", py_attention_program_build, METH_VARARGS,
     "attention_program_build(settings, sign, f64_layout): the step-by-step program's configuration from the settings."},
    {"attention_program", py_attention_program, METH_VARARGS,
     "attention_program(q, k, v, config, E, rows, kr0, vr0, S, Ev, causal, mask, mask_layout, layout, rnd, ovf, lse_layout, lrnd, lovf, refs, "
     "lse_refs, rq, rk, vv, vr, ve, vunb, mrad, tables, u, sigma): the step-by-step program's bounds for rows as attention_tolerance's, the same "
     "tuple. With inputs that are earlier results: Q's and K's radii as operands, V's values and radii as 128-bit integers at 2**ve with their "
     "no-bound flags, the float mask's radii as float64 (each None where exact). tables and u: Lemma D's tables and their unit (None and 0 "
     "without them); sigma: the scale as (numerator, denominator), or None for 1/sqrt(E)."},
    {"attention_normal", py_attention_normal, METH_VARARGS,
     "attention_normal(q, k, v, compute_layout, flush, sigma, E, rows, kr0, vr0, S, Ev, causal, mask, mask_layout, layout, rnd, ovf, lse_layout, lrnd, "
     "lovf): the normal answer for rows as attention_reference's: (the outputs' encodings, logsumexp's); sigma (num, den), or None for 1/sqrt(E)."},
    {"attention_tables", py_attention_tables, METH_VARARGS,
     "attention_tables(q, k, v, sigma, E, mask, mask_layout): Lemma D's weight tables for the call's operands and float mask (mask_layout "
     "None for none or a bool mask; v None where the caller reads V otherwise), as (capsule, u, digits), or None when the kernel does not "
     "take them (a flagged value, or distances wider than its words)."},
    {"attention_reference", py_attention_reference, METH_VARARGS,
     "attention_reference(q, k, v, tables, sigma, E, u, rows, kr0, vr0, S, Ev, causal, mask, mask_layout, layout, rnd, ovf, lse_layout, lrnd, lovf, p, cap): "
     "for rows of (query row, query index, mask offset, mask step), their outputs and logsumexp, refined from p bits up to cap; status 1 for a part "
     "still undecided at the cap."},
    {"rnd", py_rnd, METH_VARARGS, "rnd(x, p, up): the dyadic x rounded to at most p bits."},
    {"expm1_end", py_expm1_end, METH_VARARGS, "expm1_end(x, p, up): e**x - 1 for the dyadic x >= 0, rounded down or up at p bits (attention_tol.c)."},
    {"add", py_add, METH_VARARGS, "add(x, y, p, up): x + y, rounded."},
    {"mul", py_mul, METH_VARARGS, "mul(x, y, p, up): x * y, rounded."},
    {"div", py_div, METH_VARARGS, "div(x, y, p, up): x / y, rounded."},
    {"sqrt", py_sqrt, METH_VARARGS, "sqrt(x, p, up): the square root of x, rounded."},
    {"ln2", py_ln2, METH_VARARGS, "ln2(p): (lo, hi) enclosing log(2), at most p bits each."},
    {"exp", py_exp, METH_VARARGS, "exp(x, p): (lo, hi, n) with exp(x) in [lo * 2**n, hi * 2**n]."},
    {"operand_scan", py_operand_scan, METH_VARARGS, "operand_scan(data, big_endian, layout, start, rows, cols, row_step, col_step, inner, outer_step, terms, wide): (top, low, any, nonfinite, main_bits) over an operand's values, without reading it."},
    {"exp_near_one", py_exp_near_one, METH_VARARGS, "exp_near_one(x, p): (lo, hi), exp(x) rounded down and up to p bits for |x| < 2**-4 (Lemma W at three words), or None."},
    {"log", py_log, METH_VARARGS, "log(m, e, p): (lo, hi) enclosing log(m * 2**e), for m > 0."},
    {"pi", py_pi, METH_VARARGS, "pi(p): (lo, hi) enclosing pi, at most p bits each."},
    {"cos_sin", py_cos_sin, METH_VARARGS, "cos_sin(x, p): ((lo, hi), (lo, hi)) enclosing cos(x) and sin(x), at most p bits each."},
    {"cos_sin_range", py_cos_sin_range, METH_VARARGS,
     "cos_sin_range(lo, hi, p): ((lo, hi), (lo, hi)) enclosing cos and sin over [lo, hi], at most p bits each."},
    {"float_layout", py_float_layout, METH_VARARGS,
     "float_layout(exp_bits, man_bits, special, bias): (bias, itemsize, emin, emax, sign_bit), or 'bits', 'bytes', 'size' or 'range' for an "
     "invalid definition."},
    {"float_decode", py_float_decode, METH_VARARGS, "float_decode(layout, bits): (kind, negative, significand, exponent)."},
    {"float_encode", py_float_encode, METH_VARARGS,
     "float_encode(layout, kind, negative, m, e, rounding, overflow): the encoding of a value, finite ones (-1)**negative * m * 2**e."},
    {"float_encode_ratio", py_float_encode_ratio, METH_VARARGS,
     "float_encode_ratio(layout, negative, a, b, rounding, overflow): the encoding of (-1)**negative * a / b."},
    {"float_special", py_float_special, METH_VARARGS, "float_special(layout, which, negative): a zero, infinity, NaN or largest finite encoding."},
    {"float_narrow", py_float_narrow, METH_VARARGS,
     "float_narrow(data, big_endian, wide, narrow, rounding, overflow): wide-format elements rounded to the narrow format, little-endian."},
    {"int_range", py_int_range, METH_VARARGS, "int_range(itemsize, signed): (min, max), or 'size' for an itemsize over 8."},
    {"int_decode", py_int_decode, METH_VARARGS, "int_decode(bits, itemsize, signed): the integer of an encoding."},
    {"int_encode", py_int_encode, METH_VARARGS, "int_encode(n, itemsize, signed, overflow): the encoding of n, or None when out of range with 'error'."},
    {"growth_terms", py_growth_terms, METH_VARARGS,
     "growth_terms(layout, rounding, depth, roundings, w, flush): Lemma 2's (gamma, constant, largest), or None when D u >= 1."},
    {"radius_packed", py_radius_packed, METH_VARARGS,
     "radius_packed(layout, values, big, f64_layout, radii, big, count, emin, e0, lo, hi): a call packing an earlier result's values and radii "
     "lo to hi - 1 as 128-bit integers at the exponent of all count of them (e0, or None for its first pass) for the attention kernels, in tiles "
     "(radius_packed_next, radius_packed_tile, radius_packed_result)."},
    {"radius_packed_next", py_radius_packed_next, METH_VARARGS, "radius_packed_next(call): the tiles of the next pass, 0 when done."},
    {"radius_packed_tile", py_radius_packed_tile, METH_VARARGS, "radius_packed_tile(call, t): runs tile t of the current pass."},
    {"radius_packed_result", py_radius_packed_result, METH_VARARGS, "radius_packed_result(call): (values, radii, e0, unbounded)."},
    {"radius_parts", py_radius_parts, METH_VARARGS,
     "radius_parts(layout, values, big, f64_layout, radii, big, start, count, emin): earlier results' radii as float64 bytes, rounded up."},
    {"marks", py_marks, METH_VARARGS, "marks(layout, f64_layout, values, big, low, big, tolerance, big, flags, components): Bounds.marks per part."},
    {"covers", py_covers, METH_VARARGS, "covers(layout, f64_layout, output, big, low, big, high, big, flags, components): Bounds.covers per part."},
    {"recode", py_recode, METH_VARARGS,
     "recode(data, big_endian, from_layout, to_layout): each part re-encoded in to_layout's format, little-endian: rounded to nearest even "
     "with IEEE overflow for a float format, wrapped for an integer one (exact where the format holds the value)."},
    {"scalars", py_scalars, METH_VARARGS,
     "scalars(data, big_endian, layout, components): the elements as a list of Python scalars, bools for layout None and complex numbers for "
     "components 2, as each format's to_scalar gives them (NaN the quiet NaN with its sign); NotImplemented for a float format where a C "
     "double is not binary64."},
    {"nonzero", py_nonzero, METH_VARARGS,
     "nonzero(data, big_endian, layout, components, every): whether every element (every) or some element is nonzero by its encoding, NaN "
     "included; a complex element where either part is."},
    {"holds", py_holds, METH_VARARGS, "holds(f_layout, g_layout): whether format f represents every value of format g, both integer or both float."},
    {"binary_holding", py_binary_holding, METH_VARARGS, "binary_holding(bits, low, high): (exponent_bits, mantissa_bits) of a format for exact scalars."},
    {"widest", py_widest, METH_VARARGS,
     "widest(layout, center, big_endian, f64_layout, found, components): any_of's (tolerance, low, high, flags) bytes from each result's "
     "(tolerance, big, low, big, high, big, flags)."},
    {"final_bounds", py_final_bounds, METH_VARARGS,
     "final_bounds(layout, f64_layout, rounding, overflow, lo, hi, reference, w, possible): Lemma 3's (tolerance, low, high, flags)."},
    {"int_bounds", py_int_bounds, METH_VARARGS,
     "int_bounds(itemsize, signed, overflow, f64_layout, lo, hi, reference): an integer output's (tolerance, low, high) for exact values in [lo, hi]."},
    {"lemma1_ends", py_lemma1_ends, METH_VARARGS, "lemma1_ends(round item, flush, lo, hi, w): Lemma 1 on [lo, hi], or None (for the tests)."},
    {"accumulation_error", py_accumulation_error, METH_VARARGS,
     "accumulation_error(accumulation item, n, mass, w, flush): an accumulation's error bound, or None (for the tests)."},
    {"ivl_op", py_ivl_op, METH_VARARGS, "ivl_op(op, a, b, p): an interval operation's result (for the tests)."},
    {"attention_carry", py_attention_carry, METH_VARARGS,
     "attention_carry(round, accumulation, scale, flush, w, blocks): Lemma C's enclosure of carried blocks, or None (for the tests)."},
    {"matrix_exp_new", py_matrix_exp_new, METH_VARARGS,
     "matrix_exp_new(data, layout, big_endian, count, n, parts, L, box): a batch of interval matrix exponentials, each nonzero "
     "entry part widened by box (or None) (for the tests)."},
    {"matrix_exp_rational", py_matrix_exp_rational, METH_VARARGS,
     "matrix_exp_rational(data, layout, big_endian, n, parts, wanted): each wanted entry's exact value (num, den), or None."},
    {"matrix_exp_reference", py_matrix_exp_reference, METH_VARARGS,
     "matrix_exp_reference(a, layout, big_endian, count, n, parts, e, e_big_endian, out_layout, rounding, overflow, start, cap): matrix_exp's "
     "reference."},
    {"matrix_exp_reference_next", py_matrix_exp_reference_next, METH_VARARGS, "matrix_exp_reference_next(reference): the next phase's tiles, 0 when done."},
    {"matrix_exp_reference_tile", py_matrix_exp_reference_tile, METH_VARARGS, "matrix_exp_reference_tile(reference, t): runs tile t."},
    {"matrix_exp_reference_result", py_matrix_exp_reference_result, METH_VARARGS,
     "matrix_exp_reference_result(reference): (encodings, entry flags, entry midpoints, statuses) as bytes."},
    {"matrix_exp_tolerance", py_matrix_exp_tolerance, METH_VARARGS,
     "matrix_exp_tolerance(a, layout, big_endian, count, n, parts, e, e_big_endian, ra, ra_big, re, re_big, f64_layout, compute_layout, flush, "
     "allowance, out_layout, rounding, overflow, refs): matrix_exp's tolerance."},
    {"matrix_exp_tolerance_next", py_matrix_exp_tolerance_next, METH_VARARGS, "matrix_exp_tolerance_next(tolerance): the next phase's tiles, 0 when done."},
    {"matrix_exp_tolerance_tile", py_matrix_exp_tolerance_tile, METH_VARARGS, "matrix_exp_tolerance_tile(tolerance, t): runs tile t."},
    {"matrix_exp_tolerance_result", py_matrix_exp_tolerance_result, METH_VARARGS,
     "matrix_exp_tolerance_result(tolerance): (tolerances, lows, highs, flags) as bytes."},
    {"matrix_exp_next", py_matrix_exp_next, METH_VARARGS, "matrix_exp_next(batch): the tiles of the next phase, 0 when done."},
    {"matrix_exp_tile", py_matrix_exp_tile, METH_VARARGS, "matrix_exp_tile(batch, t): runs tile t of the current phase."},
    {"matrix_exp_ends", py_matrix_exp_ends, METH_VARARGS, "matrix_exp_ends(batch, m, part, i, j): an entry's ends, or None."},
    {"operand", py_operand, METH_VARARGS,
     "operand(data, big_endian, layout, start, rows, cols, row_step, col_step, inner, outer_step, terms, wide[, given]): the read kernel's "
     "operand for products of terms terms, rows in groups of inner each outer_step elements on, with the wide main range (two-word slots), "
     "as a capsule; given, a larger operand's (top, flagged) from operand_scan, reads this part as in the whole."},
    {"operand_rows_top", py_operand_rows_top, METH_VARARGS, "operand_rows_top(data, big_endian, layout, start, cols, row_step, col_step, inner, outer_step, r0, r1): rows r0..r1's (any, top bit)."},
    {"operand_new", py_operand_new, METH_VARARGS, "operand_new(layout, rows, cols, terms, wide, any, top): an operand to fill (operand_fill)."},
    {"operand_fill", py_operand_fill, METH_VARARGS, "operand_fill(x, data, big_endian, layout, start, row_step, col_step, inner, outer_step, r0, r1): rows r0..r1 read into x; (width, flagged)."},
    {"operand_done", py_operand_done, METH_VARARGS, "operand_done(x, width, flagged, given_flags): the filled operand's totals."},
    {"mma_terms", py_mma_terms, METH_VARARGS,
     "mma_terms(products, fraction_bits, exponent_floor, split, reduction_layout, slack, K, extra, steps, w, alpha=None): Lemma T's (G, H); steps "
     "-1 for the method's count; with alpha, H times max(1, |alpha|) (gemm's alpha P)."},
    {"order_depth", py_order_depth, METH_VARARGS, "order_depth(kind, k, b): the most additions on a term's path in a named order (1 sequential, 2 pairwise, 3 blocked)."},
    {"scan_flagged", py_scan_flagged, METH_VARARGS, "scan_flagged(top, low, any, nonfinite, main_bits): whether a read at main_bits flags a slot."},
    {"operand_parts", py_operand_parts, METH_VARARGS,
     "operand_parts(b, coefficients): a complex B's transposed operand (rows 2 j and 2 j + 1 the parts of column j) as the real operands "
     "of its output's real and imaginary parts."},
    {"interleave", py_interleave, METH_VARARGS, "interleave(x, y, width): the width-byte elements of x and y in turn."},
    {"pair_flags", py_pair_flags, METH_VARARGS, "pair_flags(real, imaginary): a complex output's flags from its parts'."},
    {"gather", py_gather, METH_VARARGS,
     "gather(address, shape, strides, itemsize): the itemsize-byte elements at address, of these sizes and element strides, in row-major order."},
    {"dlpack_open", py_dlpack_open, METH_VARARGS,
     "dlpack_open(capsule): an unused DLPack capsule's tensor, taken: (owner, (major or None, code, bits, lanes, shape, strides or None, "
     "address)); the owner holds it until dlpack_close."},
    {"dlpack_close", py_dlpack_close, METH_VARARGS, "dlpack_close(owner): hand a dlpack_open tensor back to its producer."},
    {"fenv_save", py_fenv_save, METH_NOARGS, "fenv_save(): the calling thread's floating-point environment as bytes, or None."},
    {"fenv_reset", py_fenv_reset, METH_NOARGS, "fenv_reset(): reset the calling thread's floating-point environment to the default; whether it worked."},
    {"fenv_restore", py_fenv_restore, METH_VARARGS, "fenv_restore(saved): set the calling thread's floating-point environment back to fenv_save's."},
    {"product_reference", py_product_reference, METH_VARARGS,
     "product_reference(a, b, a_bases, b_bases, m, n, o0, o1, layout, rounding, overflow, elementwise, epilogue): (encodings, entry flags, "
     "entry midpoints) of output elements o0 to o1, or (None, position, value) for an integer output outside its format with overflow "
     "'error'; epilogue (alpha, beta, c, big_endian, c_layout, c_terms, c_radius, c_marks, f64_layout, part) for gemm, else None."},
    {"product_int_bounds", py_product_int_bounds, METH_VARARGS,
     "product_int_bounds(a, b, a_bases, b_bases, m, n, o0, o1, layout, f64_layout, overflow, references, radii, epilogue): an integer "
     "product's (tolerances, lows, highs, flags) of output elements o0 to o1; epilogue gemm's, as product_reference's, else None."},
    {"product_normal", py_product_normal, METH_VARARGS,
     "product_normal(a, b, a_bases, b_bases, m, n, o0, o1, compute_layout, flush, layout, rounding, overflow, epilogue, exact_terms): the "
     "normal answer's encodings of output elements o0 to o1, with gemm's epilogue (product_reference's) or None; exact_terms as "
     "product_bounds takes it."},
    {"product_run", py_product_run, METH_VARARGS,
     "product_run(a, b, a_bases, b_bases, m, n, o0, o1, acc_layout, acc_rounding, method, block, fma, layout, rounding, overflow): a named "
     "algorithm's output encodings of elements o0 to o1."},
    {"product_bounds", py_product_bounds, METH_VARARGS,
     "product_bounds(a, b, a_bases, b_bases, m, n, o0, o1, compute_layout, flush, normal, exact_terms, layout, f64_layout, rounding, overflow, "
     "references, row_marks, column_marks, w, epilogue, order, mma, few, radii, reach): (tolerances, lows, highs, flags) of output elements o0 "
     "to o1; exact_terms 0 (terms rounded), 1 (values of the compute format) or 2 (exact values it need not hold); order (kind, "
     "block) for a named algorithm (kind 1 sequential, 2 pairwise, 3 blocked, 4 exact accumulation, 5 the 16-slot "
     "algorithm), mma (G, H, reduction_layout[, G', H']) for Lemma T, else None; few for the few-term rule; radii (ra, rb), the operands' "
     "radii read as operands (None where exact), or None; reach, a coefficient product's A marks per slot, or None."},
    {"mma_extent", py_mma_extent, METH_VARARGS,
     "mma_extent(data, big_endian, layout, f64_layout, radii, radii_big_endian): an operand's extent for Lemma T's slack, (may be subnormal, "
     "largest (m, e))."},
    {"mma_slack", py_mma_slack, METH_VARARGS, "mma_slack(emin, sub_x, max_x, sub_y, max_y): Lemma T's sigma (m, e) from two extents."},
    {"read_operand", py_read_operand, METH_VARARGS,
     "read_operand(data, big_endian, layout, rows, cols, row_step, col_step, main_bits): (exponent, width, slots, flagged, summary, "
     "special_rows, flagged_count), the read kernel's operand; above 62 main bits each slot is two little-endian words, low first."},
    {"exact_entry", py_exact_entry, METH_VARARGS, "exact_entry(kind, negative, m, e): (flag, midpoint) of a value."},
    {"ratio_entry", py_ratio_entry, METH_VARARGS, "ratio_entry(negative, num, den): (flag, midpoint) of (-1)**negative * num / den."},
    {"interval_entry", py_interval_entry, METH_VARARGS, "interval_entry(lo, hi): (flag, midpoint) of a value in [lo, hi], or None if too wide."},
    {"entry_read", py_entry_read, METH_VARARGS, "entry_read(flag, midpoint): (kind, negative, m, e, radius_m, radius_e, out_of_range)."},
    {"entry_bounds", py_entry_bounds, METH_VARARGS, "entry_bounds(flag, midpoint): (lo, hi), or None for NaN and infinities."},
    {"round_integer", py_round_integer, METH_VARARGS, "round_integer(num, den, rounding): num / den rounded to an integer, den > 0."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "exactref._c", NULL, -1, methods, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__c(void) {
    little_obj = PyUnicode_FromString("little");
    word_bits_obj = PyLong_FromLong(64);
    from_bytes_obj = PyObject_GetAttrString((PyObject *)&PyLong_Type, "from_bytes");
    if (!little_obj || !word_bits_obj || !from_bytes_obj) return NULL;
    PyObject *m = PyModule_Create(&module);
    if (!m) return NULL;
    /* the stored range of exact values, an entry's midpoint width and significand bits, and the
       flags of an output's non-finite values */
    const char *names[] = {"LIMIT", "ENTRY_WIDTH", "ENTRY_BITS", "FLAG_PLUS_INF", "FLAG_MINUS_INF", "FLAG_NAN", "FLAG_ANY"};
    long long values[] = {EXACT_LIMIT, ENTRY_WIDTH, ENTRY_BITS, FLAG_PLUS_INF, FLAG_MINUS_INF, FLAG_NAN, FLAG_ANY};
    for (int i = 0; i < 7; i++) {
        PyObject *v = PyLong_FromLongLong(values[i]);
        if (!v || PyModule_AddObject(m, names[i], v)) {
            Py_XDECREF(v);
            Py_DECREF(m);
            return NULL;
        }
    }
    return m;
}
