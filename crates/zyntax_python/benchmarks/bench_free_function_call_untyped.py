# `bench_free_function_call.py` with nothing annotated. The parameter
# types come from the call sites, so this should cost what the typed
# kernel costs; a gap between the two is inference falling short.
# Returns 350000000.

def step(acc, i):
    return acc + (i % 8)

def main():
    total = 0
    i = 0
    while i < 100000000:
        total = step(total, i)
        i += 1
    return total

print(main())
