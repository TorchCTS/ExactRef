# sum: proof

`exactref.sum(x, axis, keepdims=..., dtype=...)` adds the elements of `x` over
the reduced dimensions. This file proves that its reference output is the
exact sum rounded once, that `tolerance()` bounds normal accumulation and
`tolerance(standard="any_order")` every order, and that `normal()` is normal
accumulation in index order.

## sum is one matmul call

Let the outputs be indexed by o and the reduced elements of output o, in
row-major order, be x_(o,1) .. x_(o,k). `sum` builds the matrix X with rows
X[o, q] = x_(o,q) and the column of ones u (k x 1), and calls matmul's product
once: output o is sum_q X[o, q] * 1. So `matmul.md`, with the shared lemmas of
`foundations.md`, applies as written: the
exact output (The exact output), the reference output rounded once (The
reference output), the special values, the normal answer (Lemma E) and
results as inputs (an earlier result's radius r_(o,q) gives the deviation
sum_q r_(o,q) * 1). The one change is the tolerance, which uses Lemma U
(`foundations.md`) because every term x * 1 is an input value.

Integer and bool inputs are summed in int64, as PyTorch sums them: the exact
sum wraps once to the output, so the order of additions does not matter and the
tolerance is 0. A complex sum is two real sums of the same kind, one for the
real parts and one for the imaginary parts. Each part is bounded below, and the
complex tolerance holds the two parts' bounds (Lemma 3).

When a compute format is given that does not hold the input format, the
terms are rounded and matmul's bound applies unchanged.

## The backward

`grad_x` holds at every element of x the gradient of the output it was summed
into: one product per element, grad * 1, so it is exact when `grad` has x's
dtype and rounded once otherwise (an exact zero keeps grad's sign, since no
reduction happens); its tolerance is that product's (`foundations.md`, Lemma
F, or matmul's with an uncertain `grad`).
