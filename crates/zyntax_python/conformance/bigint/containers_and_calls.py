# zypy: bigint
import math

a = [1, 2, 3]
a.append(2 ** 70)
print(a, sum(a), max(a), min(a), sorted(a, reverse=True))
print(sorted([2 ** 80, 5, 2 ** 65, -2 ** 70, 7]))
print(min(2 ** 80, 2 ** 79), max([1, 2 ** 100, 3]))
d = {2 ** 80: "big", 7: "small"}
print(d[2 ** 80], d[7], (2 ** 80) in d, 5 in d)
s = {2 ** 70, 2 ** 70, 3}
print(len(s))
print(sum(2 ** i for i in range(80)), sum([2 ** 63, 2 ** 63]), sum([2 ** 62] * 4, 1))
print(pow(7, 200), pow(7, 200, 10 ** 9 + 7), pow(3, 10 ** 6, 1000003))
print(int("-98765432109876543210"), int(1e30), int(-2.5e20), int("ff" * 10, 16), int("  42  "))
print(int("123456789012345678901234567890") + 1, str(2 ** 90), len(str(3 ** 200)), repr(2 ** 65))
print(f"{3 ** 150}", f"{2 ** 70:>30}", "{}".format(2 ** 70), str(-2 ** 70))
print(hash(2 ** 70) == hash(2 ** 70), isinstance(3 ** 50, int), isinstance(5, int), isinstance(2.5, int))
print(math.factorial(25), math.gcd(2 ** 80, 6 ** 40), math.isqrt(10 ** 40 + 12345), math.comb(100, 50))
print(abs(-2 ** 70), abs(-5), divmod(2 ** 70, 3), divmod(7, 2))
xs = [10 ** 18, 10 ** 18]
xs[0] += 10 ** 18 * 20
xs[1] *= 100
print(xs)
counts = {}
for word in ["a", "b", "a"]:
    counts[word] = counts.get(word, 1) * 10 ** 12
print(sorted(counts.items()))
