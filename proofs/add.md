# add: proof

`exactref.add(x, y, alpha=a)` computes x + a * y elementwise with
broadcasting. This file proves that its reference output is the exact value
rounded once and that `tolerance()` bounds every correct implementation.

## add is one matmul call

Broadcast x and y to the output shape. Each output element is the dot product
of the row [x, y] with the column [1, a] (k = 2): the terms are x * 1 and
y * a. `add` makes all of them in one call of matmul's product. So
`matmul.md`, with the shared lemmas of `foundations.md`, applies: the exact
output and the reference output (x + a y, exactly, rounded once; its
elements are not reductions, so an exact zero takes the sign IEEE arithmetic
gives x + a y), the normal
answer (Lemma E: fl(fl(x * 1) + fl(a * y)) in the compute format, then
rounded to the output, x * 1 exact where the compute format holds x's
format; with a = 1 or -1, fl(x + a y), no term rounded on its own), results
as inputs (the deviation r_x + |a| r_y) and the tolerance:

- the model rounds a * y to the compute format, or merges it with a fused
  multiply-add, and rounds the sum once: Lemmas S and A with two terms (x * 1
  is rounded too where the compute format does not hold x's format, as
  matmul's model rounds every product);
- with a = 1 or -1 and a compute format that holds the array operands'
  format (the default compute format always does), neither term is rounded
  on its own: x and a y are exact values and only their sum is rounded, so
  the tighter bound of Lemma U (`foundations.md`) applies, its additions
  with the absolute part a where the compute format does not hold the format
  x and y are read in (a scalar operand or an earlier result of shape ()
  wider than it, taken by value); a compute format narrower than the arrays'
  rounds both terms, as matmul's model rounds every product it does not
  hold;
- an element with exact inputs and no flushing has at most two nonzero terms
  per part when a is real, so Lemma F (`foundations.md`) gives each part's
  exact worst case (with a complex a, each part that has at most two); with a = 1
  or -1 the cases are x + a y rounded once to the compute format or not at
  all, then rounded to the output.

An implementation that rounds a scalar operand to the compute format first
is `add_held_scalar("compute")` (Held scalars below).

`a` is held exactly in a format that represents it, and operands of
different formats are promoted to one that holds both values exactly, so
neither changes the exact value. A scalar operand (as in `multiply.md`: a
Python scalar, or an array of shape () of any format beside an operand with
dimensions) is taken by its value and held exactly in the narrowest format
that represents it; where the arrays' format does not, x and y are read in
one that holds both, and an integer scalar that no standard format holds with
them joins its coefficient (the term 1 times a s, exactly). Integer
sums wrap once. A complex a makes each part a sum of up to three nonzero
terms, bounded by Lemmas S and A. The column [1, a] holds exact
coefficients, so for complex values a zero part of a nonzero coefficient
adds no term to the special values (`matmul.md`, The reference output): each
part of x + a y follows IEEE rules on its own, as PyTorch's add does, and
a = 0 multiplies an infinity into NaN, as PyTorch's does.

An implementation that stores a * y in a narrower format before the addition
(PyTorch does in bfloat16, and in float16 when y is a broadcast row) makes a
rounding the model does not have; it is the chain add(x, multiply(y, a,
dtype=...)), whose bound comes from results as inputs.

## Held scalars

`add_held_scalar(held)` describes an implementation that first rounds each
scalar operand and a to nearest even to a format H, or to each format of a
sequence in turn, as `multiply.md` (Held scalars) describes, then computes
x + fl_H(a) y as the model above does, with each scalar's held value in its
place: Lemma U where fl_H(a) is 1 or -1, Lemma F's cases, or Lemmas S and
A. The held values are known exactly, so the interval before the final
rounding comes from them, and the bounds are measured from add's reference
with the scalars and a as given, by Lemma 3, as in `multiply.md`; `normal()`
is the normal answer at the held values. An implementation that also stores
fl_H(a) y in a narrower format first (as PyTorch's bfloat16 kernel does) is
the chain above with a so rounded.

## The backward

grad_x is grad summed over x's broadcast copies, and grad_y is conj(a) * grad
summed over y's. Each is one matmul call whose rows are the copies and whose
column holds 1 or conj(a), so each element is one exact sum rounded once and
its tolerance is matmul's (`foundations.md`'s Lemma U with a column of ones or
of -1). An element
with one copy, of an operand with the output's number of dimensions, is a
single product, not a reduction, so an exact zero takes that product's IEEE
sign. Any other element keeps the sum rule (`matmul.md`, The reference output):
PyTorch sums it, including over the dimensions broadcasting adds in front,
even of size 1.
