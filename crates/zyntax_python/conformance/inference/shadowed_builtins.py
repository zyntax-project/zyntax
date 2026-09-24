# A parameter, local or global named like a builtin is called through
# the value it holds, so the call's result is that function's.
def pairs_len(ps):
    return len(ps)


def run(dict):
    return dict([(1, 2), (3, 4)])


def run2(list):
    return list([1, 2])


def run3(sorted, tuple):
    return sorted([3, 1]), tuple([5])


def run4(xs):
    sum = pairs_len
    return sum(xs) + 1


print(run(pairs_len), run2(pairs_len), run3(pairs_len, pairs_len), run4([7, 8]))
print(run(lambda ps: ps[0]), run2(lambda xs: xs[-1]))
min = pairs_len
print(min([4, 5, 6]) * 2)
