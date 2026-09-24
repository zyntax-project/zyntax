# A structure built and finished with before a long loop in the same
# call: once the frame leaves for compiled code at the loop, nothing
# holds it, and a collection during the loop must not reach it.
def build(n):
    xs = []
    for i in range(n):
        xs.append([i, i + 1])
    return xs


def run(n, steps):
    big = build(n)
    count = len(big)
    keep = []
    total = 0
    i = 0
    while i < steps:
        keep.append([i])
        if len(keep) == 256:
            keep = []
        total += i & 7
        i += 1
    return total + count


print(run(200000, 3000000))
