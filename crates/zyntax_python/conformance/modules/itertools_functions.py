import itertools
from itertools import accumulate, chain, combinations, count, cycle, groupby, islice, permutations, product, repeat, zip_longest

print(list(chain([1, 2], (3,), "ab")), list(chain()), list(chain.from_iterable([[1], [2, 3], []])))
print(list(islice(count(5), 3)), list(islice(count(10, -2), 4)), list(islice("abcdef", 2, 5)), list(islice(range(10), 1, 8, 3)))
print(list(islice(cycle("xy"), 5)), list(repeat("z", 3)), list(islice(repeat(7), 2)))
print(list(accumulate([1, 2, 3, 4])), list(accumulate([1, 2, 3], lambda a, b: a * b)), list(accumulate([1, 2], initial=10)))
print(list(itertools.takewhile(lambda x: x < 3, [1, 2, 3, 1])), list(itertools.dropwhile(lambda x: x < 3, [1, 2, 3, 1])))
print(list(itertools.filterfalse(lambda x: x % 2, range(6))), list(itertools.compress("abcd", [1, 0, 1, 0])))
print(list(itertools.starmap(lambda a, b: a + b, [(1, 2), (3, 4)])))
print(list(zip_longest([1, 2, 3], "ab")), list(zip_longest([1], [2, 3], fillvalue=0)))
print(list(product([1, 2], "ab")), list(product([0, 1], repeat=2)), list(product("a")), len(list(product(range(3), range(4)))))
print(list(permutations([1, 2, 3])), list(permutations("abc", 2))[:4], len(list(permutations(range(4)))))
print(list(combinations([1, 2, 3, 4], 2)), list(combinations("abc", 3)), list(combinations([1, 2], 3)))
print(list(itertools.combinations_with_replacement("ab", 2)))
for key, group in groupby("aabbbc"):
    print(key, list(group))
print([(k, len(list(g))) for k, g in groupby([1, 1, 2, 3, 3, 3])], [k for k, g in groupby([5, 15, 25, 7], lambda x: x // 10)])
print(list(itertools.pairwise([1, 2, 3, 4])), list(itertools.batched("abcdefg", 3)))
total = 0
for a, b in zip(list(islice(count(1), 3)), "xyz"):
    total += a
print(total)
it = islice(count(), 3)
print(next(it), next(it), next(it))
print(sum(islice(count(1), 100)), max(islice(cycle([3, 9, 4]), 7)))
