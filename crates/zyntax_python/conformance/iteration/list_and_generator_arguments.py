def ls(x):
    return list(x)

def total(xs):
    return sum(xs)

def collect(xs):
    out = []
    for v in xs:
        out.append(v * 2)
    return out

def gen():
    yield 1
    yield 2
    yield 3

def squares(n):
    for i in range(n):
        yield i * i

print(ls([1]), ls(range(2)), ls(x for x in [4, 5]), ls(gen()), ls(squares(4)))
print(total([1, 2]), total(range(3)), total(x for x in [4, 5]), total(gen()), total(squares(4)))
print(collect([1]), collect(range(2)), collect(x + 1 for x in [4, 5]), collect(gen()))
print(ls("ab"), ls({1: 2}), ls((7, 8)), ls({9}))

def mk(flag):
    return (x for x in [1, 2]) if flag else [3]
g = mk(True)
for v in g:
    print(v)
print(list(mk(True)), list(mk(False)))
items = [(x for x in [1, 2]), [3]]
for it in items:
    print(list(it))
print(type(x for x in [1]).__name__)
print(sorted(ls(y for y in [3, 1, 2])))
print(max(ls([1, 5, 2])), max(ls(z for z in [4, 9, 2])))
print("-".join(ls(s for s in ["a", "b"])), "-".join(ls(["c", "d"])))
