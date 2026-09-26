# A large dict: 200,000 int keys spread over a wide range, built once,
# then probed twice as many times with hits and misses mixed.
# Returns 2440003.

def main() -> int:
    d = {}
    for i in range(200000):
        d[(i * 2654435761) % 4294967296] = i
    hits = 0
    s = 0
    for i in range(400000):
        k = (i * 2654435761) % 4294967296
        if k in d:
            hits += 1
            s += d[k]
    return hits * 7 + s % 1000003 + len(d)

print(main())
