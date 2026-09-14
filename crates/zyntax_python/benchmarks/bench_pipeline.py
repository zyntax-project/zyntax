# A comprehension pipeline: a list built by comprehension, filtered by
# another, reduced through a generator expression, paired into tuples,
# sorted by a key lambda and unpacked. Every stage is a Python idiom
# that costs a fiber, a box or a dynamic call unless it is fused away.
# Returns 867144200.

def main() -> int:
    n = 1500000
    xs = [(i * 2654435761) % 1000003 for i in range(n)]
    evens = [x for x in xs if x % 2 == 0]
    squares = sum(x * x % 1000 for x in evens)
    pairs = [(x % 1000, x) for x in evens[:200000]]
    pairs.sort(key=lambda p: p[0])
    acc = squares
    for k, v in pairs[:1000]:
        acc += k * 31 + v
    return acc % 1000000007

print(main())
