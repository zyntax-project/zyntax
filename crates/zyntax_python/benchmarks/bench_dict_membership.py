# Membership: `in` on a set and on a dict, hits and misses, with int
# and string members.
# Returns 509100.

def main() -> int:
    ints = set()
    for i in range(0, 60000, 3):
        ints.add(i)
    names = {}
    for i in range(5000):
        names["k" + str(i * 11)] = i
    probes = []
    for i in range(0, 2000):
        probes.append("k" + str(i * 7))
    found = 0
    for r in range(25):
        for i in range(60000):
            if i in ints:
                found += 1
        for p in probes:
            if p in names:
                found += 2
    return found

print(main())
