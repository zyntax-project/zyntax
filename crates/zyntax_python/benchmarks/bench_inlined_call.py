# The baseline the call benchmarks are read against: the arithmetic
# done inline, no call in the loop body. What the other kernels add to
# this number is the cost of the call shape they exercise.
#
# The accumulator steps by `i % 8` rather than a constant, so the loop
# is not an affine recurrence the pipeline folds to a closed form.
# Returns 350000000.

def main() -> int:
    total = 0
    i = 0
    while i < 100000000:
        total = total + (i % 8)
        i += 1
    return total

print(main())
