# any_of: proof

`exactref.any_of(center, *alternatives)` returns the center's result with
its bounds widened to cover an implementation that computes any one of the
alternatives' expressions instead of the center's.

**Claim.** For each element (each part of a complex element) and each set of
tolerance arguments, let [lo, hi] be the smallest range holding the finite
ranges of the center and every alternative R, F the union of their flags, and
T the largest distance from center.rounded to an end of those ranges, rounded
upward. If an implementation's output o is one that R's bounds allow for some
R, then o is in [lo, hi] when finite, its value's flag is in F when not, and
for a finite o, |o - center.rounded| <= T.

**Proof.** A finite o lies in R's finite range, inside [lo, hi], so its
distance from center.rounded is at most the distance to the farther end of
that range, which is at most T. A non-finite o has its flag in R's flags,
inside F. Each distance is exact before it is rounded up to float64. Since
R's range lies within R.rounded +- R.tolerance, T is never larger than the
largest |R.rounded - center.rounded| + R.tolerance(...). []

**Special values.** A center part that is not finite has a NaN tolerance, as
every tolerance has where the reference is not finite. Where the center part
is finite but an alternative's reference is not, the alternative adds only
the finite outputs it can still produce (none, or its range when its exact
value rounded past the largest finite value) and its flags. Where the center
or an alternative has an infinite tolerance (no bound), T is inf.
