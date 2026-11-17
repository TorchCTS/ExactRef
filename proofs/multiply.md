# multiply: proof

`exactref.multiply(x, y)` multiplies elementwise with broadcasting. This file
proves that its reference output is the exact product rounded once and that
`tolerance()` bounds every correct implementation.

## multiply is one matmul call

Broadcast x and y to the output shape. Each output element is the 1 x 1 matrix
product of its x and its y (k = 1), and `multiply` makes all of them in one
call of matmul's product, with the elements as batches. So `matmul.md`, with
the shared lemmas of `foundations.md`, applies: the exact output and the
reference output (one exact product rounded once), the normal answer (Lemma E:
the product rounded to the compute format, then to the output), results as
inputs (the deviation |x| q + r |y| + r q) and the tolerance (Lemmas S and A
with one term; for complex operands each part is the two-term sum
a_re b_re - a_im b_im or a_re b_im + a_im b_re). Integer products wrap once.
Its elements are not reductions, so an exact zero takes the sign IEEE
arithmetic gives the product, and each complex part the sign of its two-term
sum (`matmul.md`, The reference output).

Operands of different formats are promoted to one that holds both values
exactly (as PyTorch promotes). A scalar operand is taken by its value and held
exactly in a format that represents it: a Python scalar, and an array of shape
() beside an operand with dimensions, which PyTorch leaves out of the
promotion, whatever its format. Neither changes a value, so the exact product
is the same, and the model has no rounding of a scalar: its one rounding before
the output's is the product's. An integer scalar is not wrapped to the
array's format first either, so with overflow "saturate" or "error" an exact
product outside the output format saturates or raises. An earlier result of
shape () there is an operand in its own format, with its bounds
(`foundations.md`, Results as inputs).

**Special values.** Real products follow IEEE rules (matmul's rule with one
term). A complex product whose operands have a non-finite part follows IEEE
rules in each part separately, as PyTorch's elementwise product does: each part
is the IEEE sum of its two IEEE products. (Every input part appears in both
output parts, so both parts are then non-finite and their tolerance is NaN.)

The code applies Lemma F (`foundations.md`, few terms) to each part that
qualifies and keeps the general
bound for the others (an uncertain operand with a nonzero radius, flushing, or
more terms). An earlier result whose radius is 0 there qualifies too, with
its flags (`foundations.md`, Results as inputs): each stored part is then its
midpoint or a non-finite value its flags allow. With every stored part its
midpoint the inputs are exact and Lemma F gives the values. With some stored
part non-finite, every term it reaches is non-finite, and an infinity or NaN
never becomes finite again in an addition or a product (`foundations.md`), so
the value before the last rounding is non-finite: it adds no finite value and
reaches the output as the flags it passes on, which join the cases. With at
most two nonzero terms the operation itself makes an infinity only by a term
rounded on its own that overflows, or by the one sum; both are among Lemma F's
values at the midpoints, so under either standard a passed infinity gives
NaN where those values hold an infinity of the other sign. The
terms are counted after the coefficient rule (`matmul.md`, The reference
output): with exact coefficients, a zero part of a nonzero coefficient adds no
term, so the finite part next to a non-finite one can qualify. A case whose
rounding overflows gives an infinity (or NaN), whose flag is set, and the
finite cases give the tolerance and the finite range. A case whose
compute rounding overflows in a C without infinity is NaN (`foundations.md`,
Lemma 3: compute roundings keep the IEEE rule with NaN for the missing
infinity). Both standards keep every case: a product rounded on its own past
the threshold overflows in every implementation that rounds it
(`foundations.md`, Lemma O (a)). For a real product of two values of a format that C holds,
the cases are fl_O(fl_C(t_1)) and fl_O(t_1), and the tolerance is often exactly 0.

## Held scalars

`multiply_held_scalar(held)` describes an implementation that first rounds
each scalar operand s to nearest even to a format H (`held`: the array
operands' promoted format, the compute format, or a format given; a complex s
part by part), a value past H's range giving an infinity (NaN in a format
without one), as a cast does; then it computes the product as the model above
does. With a sequence of formats H_1, ..., H_m, the held value is s rounded
to each in turn, fl_H(s) = fl_Hm(... fl_H1(s)), as an implementation that
converts s through another format does (PyTorch converts a float64 value to
bfloat16 through float32). PyTorch's CPU kernels hold their scalars so (the
algorithm's docstring gives their H). The value fl_H(s) is known exactly:
Python rounds it, once per format, by the C codec. So the implementation is
this file's model on the operands x and fl_H(s), and the interval X its
value before the final rounding lies in, Lemma F's cases or Lemmas S and A
(with results as inputs for x), comes from them alone.

Lemma 3 (`foundations.md`) holds for any reference: the finite range [lo',
hi'] comes from X alone, and |output - ref| <= max(hi' - ref, ref - lo') for
every finite output, whether or not ref lies in the range (one of the two
differences is then negative). Lemma F's exact worst case is likewise the
largest distance from ref to its cases rounded to O. The bounds are measured
from multiply's reference with s as given, fl_O(x s), so the tolerance holds
the scalar's rounding, and the range need not hold the reference. The
product's bounds kernel takes that reference in place of the product's own,
and the per-element work stays in it. Where x fl_H(s) is not finite (fl_H(s)
an infinity, or 0 beside an infinite x), so is every output (`foundations.md`,
A finite part next to a non-finite input), and the value IEEE rules give it
need not be the reference's own: the kernel adds its flag, and where the
reference is finite it gives no finite bound (an infinite tolerance, every
finite output and every flag). A reference that is not finite joins the flags
as Lemma 3 adds it, which can make them wider than needed, never narrower.
`normal()` is the normal answer on x and fl_H(s).

An earlier result of shape () as s may be stored anywhere its bounds allow,
so it has no one value to round: the algorithm raises ValueError. The chain
multiply(s, 1, dtype=H) holds it in H, and multiply takes that result as an
operand in its own format, with its bounds.

## The backward

grad_x = grad * conj(y), summed over the copies that broadcasting made of x,
in x's shape; grad_y likewise. A scalar operand y enters grad_x by its value,
as in the forward. Each is one matmul call whose rows are the
copies (the same folding as matmul's backward), so each element is one exact
sum rounded once and its tolerance is matmul's, or Lemma F's where only one
copy is summed. Where copies are summed, PyTorch may round each product to the
dtype first; that rounding is not part of this model (a chain covers it).
