# Free-function call latency, against `bench_inlined_call.py`.
#
# The loop body is one call to a module-level `def` doing the
# arithmetic the baseline does inline. `step` is small and pure, which
# is what the inliner takes, so the delta to the baseline is expected
# to be near zero: the number says the call was removed, not that
# calls are free. Returns 350000000.

def step(acc: int, i: int) -> int:
    return acc + (i % 8)

def main() -> int:
    total = 0
    i = 0
    while i < 100000000:
        total = step(total, i)
        i += 1
    return total

print(main())
