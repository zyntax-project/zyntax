class Base:
    kind = "base"

    def __init__(self):
        self.a = 1

    def run(self):
        return "run"


class Child(Base):
    def __init__(self):
        Base.__init__(self)
        self.b = 2.5

    def extra(self):
        return "extra"


class Plain:
    def __init__(self):
        self.a = "text"


def probe(o):
    return (hasattr(o, 'a'), hasattr(o, 'b'), hasattr(o, 'run'), hasattr(o, 'kind'),
            hasattr(o, 'append'), hasattr(o, '__len__'), hasattr(o, 'upper'),
            hasattr(o, 'real'), hasattr(o, 'items'), hasattr(o, 'count'))


things = [Base(), Child(), Plain(), "s", [1, 2], {1: 2}, {3}, (1, 2), 5, 2.5, None]
for t in things:
    print(probe(t))


def fetch(o):
    return getattr(o, 'a', "none"), getattr(o, 'b', -1)


for t in [Base(), Child(), Plain(), 5, None, "s"]:
    print(fetch(t))


def poke(o):
    setattr(o, 'a', 42)
    return o.a


print(poke(Base()), poke(Child()))
print(getattr(Child(), 'b'), getattr(Base(), 'a'))

try:
    getattr(Base(), 'zzz')
except AttributeError as e:
    print("AttributeError", e)

n = 0


def bump():
    global n
    n += 1
    return n


print(getattr(Base(), 'a', bump()), getattr(Base(), 'q', bump()), n)
print(getattr("ab", 'upper', None) is not None)
x = getattr(Child(), 'b', 0.0) + 1
print(x)
def call_it(o, default):
    f = getattr(o, 'run', default)
    return f()


for o in [Base(), Child()]:
    print(call_it(o, None))
print(getattr(Child(), 'extra')())
print(getattr(Base(), 'kind'), getattr(Child(), 'kind', 1))
items = [Base(), "str", [1], 5]
print([hasattr(i, 'run') for i in items])
print([getattr(i, 'a', 0) for i in items])
print(getattr("abc", 'upper')())
print(getattr([3, 1, 2], 'append', None) is not None)
o = None
print(hasattr(o, 'a'), getattr(o, 'a', 'dflt'))
