def items(t):
    for x in t:
        yield x


print(list(items((1, 2, 3))), list(items(("a", "b"))), list(items(([1], [2, 3]))))
