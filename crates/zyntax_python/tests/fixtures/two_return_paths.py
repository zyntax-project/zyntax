# One call, long enough to move to the optimizing tier mid-loop, whose
# loop reads the result of a callee with two return paths.
def signed(x):
    if x & 1:
        return x
    return -x


def run(n):
    total = 0
    i = 0
    while i < n:
        total += signed(i) + 3
        i += 1
    return total


print(run(40000000))
