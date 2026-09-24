# == and != against None, on values that cannot be None, on instances
# with and without __eq__, and on dynamic values; an operand that runs
# still runs.
calls = []


def counted(v):
    calls.append(v)
    return v


n = 5
s = "s"
xs = [1]
f = 2.5
print(n == None, n != None, s == None, s != None, xs == None, xs != None)
print(f == None, None == f, None != f, () == None)
print(counted(1) == None, counted(2) != None, None == counted(3), calls)


class Plain:
    pass


class Eq:
    def __init__(self, v):
        self.v = v

    def __eq__(self, other):
        return other is None or self.v == other.v


p = Plain()
q = Eq(1)
print(p == None, p != None, q == None, q != None)
maybe = p
for i in range(2):
    print(maybe == None, maybe != None)
    maybe = None
d = {"k": None}
for v in [d.get("k"), d.get("z", 0), None]:
    print(v == None, v != None)
