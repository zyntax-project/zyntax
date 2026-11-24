# hash() of an int is its value modulo 2 ** 61 - 1, with the value's sign.

values = [0, 1, -1, -2, 12345, -12345, (1 << 61) - 2, (1 << 61) - 1, 1 << 61, (1 << 61) + 1,
          (1 << 62) + 1, (1 << 63) - 1, -(1 << 61), -((1 << 61) - 1), -((1 << 61) + 5), -(1 << 63), 10 ** 18]
print([hash(v) for v in values])
print(hash(True), hash(False), hash(2) == hash(2), hash(-1) == hash(-2))
d = {v: i for i, v in enumerate(values)}
print(d[1 << 62 | 1], d[-(1 << 63)], len(d), sorted(set(values))[:3])
