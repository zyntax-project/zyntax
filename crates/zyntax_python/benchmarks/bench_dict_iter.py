# Iteration over a dict's items, keys and values: a 20,000-entry dict
# walked in each of the three ways, repeatedly.
# Returns 67752960.

def main() -> int:
    d = {}
    for i in range(20000):
        d[i * 3] = i % 97
    s = 0
    for r in range(60):
        for k, v in d.items():
            s += k * v % 13
        for k in d.keys():
            s += k % 7
        for v in d.values():
            s += v
    return s

print(main())
