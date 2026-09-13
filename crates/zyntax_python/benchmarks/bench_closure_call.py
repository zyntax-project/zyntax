# Calling a closure that captures a variable of its defining function.
# The captured `k` lives in a cell the closure reads on every call, on
# top of what `bench_lambda_call.py` pays. Ten million iterations.
# Returns 35000000: each run of eight consecutive `i` contributes 28.

def make_step(k):
    def step(acc, i):
        return acc + (i + k) % 8
    return step

def main() -> int:
    step = make_step(3)
    total = 0
    i = 0
    while i < 10000000:
        total = step(total, i)
        i += 1
    return total

print(main())
