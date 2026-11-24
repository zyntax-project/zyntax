# zypy: bigint
# Values at the edges of a limb and of a machine int, in both signs.

edges = []
for k in (58, 59, 60, 61, 62, 63, 64, 65, 119, 120, 121, 122, 127, 128, 180, 181):
    for d in (-1, 0, 1):
        edges.append((1 << k) + d)
        edges.append(-((1 << k) + d))

total = 0
for x in edges:
    if x > (1 << 63) or x < -(1 << 63):
        total += hash(x) % 1000003
    assert int(str(x)) == x
print(total, len(edges))
print(sum(len(str(x)) for x in edges))

pairs = [(edges[i], edges[(i * 7 + 3) % len(edges)]) for i in range(len(edges))]
sums = []
for a, b in pairs:
    sums.append(a + b)
    sums.append(a - b)
    sums.append(a * b)
    if b != 0:
        sums.append(a // b)
        sums.append(a % b)
        q, r = divmod(a, b)
        assert q * b + r == a
print(sum(x % 998244353 for x in sums), len(sums))
print(max(sums) > 0, min(sums) < 0, sorted(set(edges))[:3] == sorted(set(edges))[:3])

m = -(1 << 63)
print(m, m - 1, m * -1 == 1 << 63, abs(m), m % 7, (m + 1) // 2, divmod(m, -1))
print((1 << 64) - 1 == 18446744073709551615, (1 << 64) >> 3, (1 << 120) >> 60, (1 << 120) >> 121)
print(((1 << 130) | 12345) & ((1 << 70) - 1), ((1 << 130) ^ (1 << 129)) >> 128, ~(1 << 100), (-(1 << 100)) & 0xffff)
print(float(1 << 100), float(-(1 << 70)), int(float(1 << 80)), (1 << 100) / (1 << 98))
print(pow(3, 200, 10 ** 9 + 7), pow(2, 1000, (1 << 89) - 1), 7 ** 90 % (1 << 70))
print((10 ** 18) * (10 ** 18), str(10 ** 36), str(10 ** 36 + 1), str(10 ** 54 - 1), str(3 * 10 ** 17))
print(int("1" + "0" * 40) // (10 ** 18), int("9" * 45) % (10 ** 18 + 9))
