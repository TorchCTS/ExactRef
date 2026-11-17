# The kernels' ranges

The kernels in `csrc/` compute on fixed-width words. For each kernel, this file
shows that every word it fills holds its value exactly: no sum, product or shift
leaves its word. The algorithms' bounds are proved in the operations' own files
and in `foundations.md`; the kernels only have to compute their quantities
exactly. Notation as in `foundations.md`: top(x) is the integer with
2**(top - 1) <= |x| < 2**top.

## Lemma R: the read kernel

`read_operand` reads an operand's values, in a float format of at most 64 bits,
for the product kernels: integers at one main exponent, with the values below
that range kept in place as flagged slots. Let W <= 126 be the main range's bits
and t the largest top(v) over the operand's finite nonzero values v. The main
exponent is E = t - W (0 when there is no such value). For W <= 62 each slot is
one 64-bit word; above 62 it is two, a signed 128-bit integer.

Write a finite nonzero value v = (-1)**s m 2**f with m odd. If f >= E, its slot
holds the integer v / 2**E = (-1)**s m 2**(f - E), and

    |v / 2**E| = |v| 2**-E < 2**t 2**-E = 2**W,

so the slot's signed word holds it exactly (W <= 62 in one word, W <= 126 in
two), and the shift that forms it, m << (f - E), moves m's top bit to bit
top(v) - E - 1 < W, inside the word. Zero is the integer 0. Every other value
(f < E, or not finite) is flagged: its slot holds the value's own encoding,
which has at most 64 bits (in the low word). So every slot holds
its value exactly, as an integer at E or as its encoding, and the bitmap says
which. A summary byte is set exactly when its block of 8 terms of its row holds
a flagged slot, and a row's special byte exactly when the row holds NaN or an
infinity.

The kernel reports the largest bit length of an unflagged slot, at most W: the
product kernels' domain checks take it as the operand's width.

## Lemma P: the product kernel

`product.c` computes, for each output element of a real product, the terms
t_q = a_q b_q (q < K) of row i of A's operand and row j of B's (column j of
B), both read by the read kernel with the main range W of K terms
(`product_main_bits`): the largest W with 2 W + 2 ceil(log2 K) + 1 <= 128, at
most 62, or in the wide main range, taken for formats of more than 32 bits of
precision and with radii, the largest W with 2 W + 2 ceil(log2 K) + 1 <= 256,
at most 126. Write c = ceil(log2 K), so K <= 2**c.

**The fast path.** Where neither row holds a flagged slot, every factor is a
slot's integer, below 2**W in magnitude (Lemma R), so |t_q| < 2**(2 W), and the
product of the two words, each below 2**63, is exact in the signed 128-bit
word. Each sum is then bounded by the mass, sum |t_q| < K 2**(2 W) <= 2**(2 W + c):
- the exact sum and every running sum (index order's, each lane's) have
  magnitude below 2**(2 W + c) <= 2**(127 - c), inside the signed word;
- the mass and the moved magnitudes are below 2**(2 W + c), as unsigned
  words;
- an order's node sum adds the magnitudes of at most K running sums (each
  lane's first running sum is its first term, counted and then taken back)
  and of the combination's nodes, each at most the mass, at most K values
  in all, so mass plus node sum is at most K times the mass, below
  2**(2 W + 2 c), and N = 2 (mass + node sum) < 2**(2 W + 2 c + 1) <= 2**128,
  inside the unsigned word.

The words' arithmetic wraps modulo 2**128; since every value it forms lies
inside its word, each is the exact integer.

In the wide main range the same holds one size up: both operands' slots are
signed 128-bit integers below 2**W <= 2**126, their product is exact in a
signed 256-bit word, every sum above lies below 2**(2 W + c) <= 2**(255 - c),
and N < 2**(2 W + 2 c + 1) <= 2**256, so the 256-bit words' arithmetic, which
wraps modulo 2**256, forms each exactly. The named orders' nodes, and elements
of one operand in the wide main range and one not, take the wide path.

**The wide path.** Where a row holds a flagged slot, each factor is read as
sig 2**e with sig odd or a slot's integer at its operand's exponent, so every
term is an integer times 2**f, f the sum of the two rows' smallest such
exponents (the lowest set bit of the row's values, or the operand's exponent
where that is lower). Its magnitude is below 2**(t - f), t the sum of the two
rows' top bits (Lemma R's top(v), over the row). The element's integers take
as many 64-bit words as hold t - f + 2 bits(K) + 3 bits, where bits(K) >= c:
every sum above is bounded as on the fast path with t - f in place of 2 W,
the signed ones with one bit more, so each fits, and the two's-complement
arithmetic on the words is exact.

**The per-block path.** Where few blocks of a row pair hold a flagged
slot, blocks without one take the fast path's words and flagged blocks the
wide path's, both for the same exact quantities. Write u = 2**(E - f), E the
main grid's exponent (the operands' exponents' sum) and f the wide path's.
Each running sum V (index order's, each lane's) is held as V = M u + rem with
M = floor(V / u) and 0 <= rem < u. A clean block's terms are multiples of u,
so they change M only, and its node is

    |V| = M u + rem = |M| u + rem      where M >= 0,
    |V| = -(M u + rem) = |M| u - rem   where M <= -1 (then V <= -u + rem < 0),

so each order's node sum over a stretch of clean blocks is u times the sum of
|M| plus rem times (the nodes with M >= 0 less those with M < 0), all exact. A
flagged block adds its terms to V in full and splits V again after it. The
ranges are the fast path's: every |V| is at most the mass, below 2**(2 W + c) u,
so |M| <= |V| / u + 1 < 2**127 fits the signed word, and a sum of at most K
values |M| is below K (2**(2 W + c) + 1) <= 2**(2 W + 2 c) + K <= 2**126 + K
(2 W + 2 c is even and at most 127), so it fits the signed word too, as
`mixed_terms` reads it. The
multi-word integers cover the wide path's bits and the unit u, so each value
in them, a remainder and M u included, is exact.

**Non-finite values.** An element whose row of A or column of B holds NaN or
an infinity takes no sums: its value is the IEEE result of its exact terms
(NaN from a NaN factor or an infinity times zero, or from infinities of both
signs; otherwise the infinity).

`sums.c` and `finish.c` take each element's integers as multi-word values,
whose arithmetic is exact, so the bounds' quantities are exact as well.

## Lemma K: the 16-slot order

`slots.c` puts each nonzero rounded product t of an element in the slot
min(15, floor(Delta / p)), Delta = floor(log2((T / |t|)**16)), T the largest
magnitude, which is `matmul.md`'s rule: 2**(s p) <= (T / |t|)**16 exactly
when s p <= Delta. Write each magnitude as mant 2**(top - 64) with mant's top
bit set (mant < 2**64, from the format's significand), and d = top_T - top_t.
Then T / |t| = (mant_T / mant_t) 2**d with mant_T / mant_t in (1/2, 2), so

    Delta = 16 d + g,   g = floor(log2((mant_T / mant_t)**16)) in [-16, 15],

and Delta >= 0 since |t| <= T. Where every Delta in [max(16 d - 16, 0),
16 d + 15] gives one slot, that is the slot; otherwise g is computed from the
integers mant_T**16 and mant_t**16 (at most 1024 bits): with
b = bits(mant_T**16) - bits(mant_t**16), g = b, less 1 where
mant_T**16 < mant_t**16 2**b (b >= 0) or mant_T**16 2**-b < mant_t**16
(b < 0). Every step is exact integer arithmetic. Within a slot the values sort
by sign, then magnitude, then index (`slots.c`'s `key_order` under `qsort`,
ties broken by the index), and the exact products, sums and node sums are
multi-word integers at the element's lowest exponent, so Lemma S's quantities
are exact.

## Lemma J: the attention gradient kernel

`attention_grad_tol.c` bounds the backward's sums
(`scaled_dot_product_attention_backward.md`, The tolerance) from intervals
whose ends are integers at their format's least unit (Values of a format)
and exact factors read by the read kernel. Take one sum of n <= 2**c terms
w x, with w in [lo, hi] an interval factor (P', dS', or for D' the product of
P' and dP') and x exact, or an interval itself (dP' in D'). Let 2**tw bound
|lo| and |hi| over the call (ends below 2**tw), and 2**tx bound |x| + rx,
rx being x's radius as an earlier result, 0 otherwise (for an interval x, its
ends' magnitudes).

The kernel sizes the P' and dS' ends' words first from P' below 4 and dS'
below 32 Ev max|g| max|v| (times |scale| with scale_ds), which most calls
keep to, but which nothing guarantees: with wide input intervals P''s
enclosure loses the correlation of its quotient, and a long row in a narrow
format can hold probabilities summing past 1. From those sizes it chooses
the sums' words (Words below), and with them the most bits an end may take:
with fixed words of B bits, the most that keep tw + tx + 2c + 6 <= B for every
sum the end joins (P' grad_value's; dS' grad_key's and grad_query's), and at
most its own words' bits; with multi-word integers, the end's words. Each row
converts its ends and reports the bits an end needs where it takes more; the
call is then made again with ends of at least those bits (`_backward.py`,
`_Wider`), and past the fixed words' most that call takes multi-word
integers. So the call that finishes has every end within the bits its sums
were sized for, and 2**tw above bounds them all.

**Integers.** The doubled midpoint a = lo + hi and radius ra = hi - lo are
below 2**(tw + 1) in magnitude, and so are b and rb for x: b = 2 x and
rb = 2 rx, or for the hull of 0 and x under flush b = x and rb = |x| + 2 rx,
or for an interval x its doubled midpoint and radius. Then

- each reference term |a b| < 2**(tw + tx + 2);
- each distance (|a| + ra)(|b| + rb) - |a| |b| = |a| rb + ra |b| + ra rb is
  below 3 2**(tw + tx + 2) < 2**(tw + tx + 4);
- for the any-order bound, each end product |lo x| or |hi x| and each
  magnitude max(|lo|, |hi|)(|x| + rx) is below 2**(tw + tx).

**Sums.** Over n terms, the exact sum, every running sum (index order's,
each lane's) and the mass are below n 2**(tw + tx + 2) <= 2**(tw + tx + 2 + c)
in magnitude, and R is below 2**(tw + tx + 4 + c). A node sum adds at most n
node magnitudes, each at most the mass, so it is below 2**(tw + tx + 2 + 2 c),
and N + 2R = 2 (mass + node sum) + 2R is below 2**(tw + tx + 2 c + 6). The
any-order sums are below 2**(tw + tx + c).

**Words.** The kernel holds a call's sums in fixed words of B bits (two's
complement for the signed ones) when tw + tx + 2 c + 6 <= B, so that every
value above lies inside its word and the words' arithmetic, which wraps
modulo 2**B, forms each exactly. Otherwise it takes multi-word integers sized
from the same bounds, whose arithmetic is exact. Each product of an end (at
most tw + 1 bits) and a factor (at most tx + 1 bits) is formed exactly in a
word of at least tw + tx + 2 bits.
