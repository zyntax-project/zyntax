# An instance compared with a value of another kind, and an instance's
# truth from __bool__, from __len__, or true by default.
class Plain(object):
    pass


class Flag(object):
    def __init__(self, on):
        self.on = on

    def __bool__(self):
        return self.on


class Bag(object):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)


class Same(object):
    def __eq__(self, other):
        return isinstance(other, Same)


def held(i):
    return [Plain(), Flag(False), Flag(True), Bag([]), Bag([1]), Same(), 3, "s"][i]


for i in range(8):
    v = held(i)
    print(i, bool(v), not v, v == 3, v == "s", v != 2.5, v == v)
    if v:
        print("true")
    else:
        print("false")
s = Same()
print(s == Same(), s == 1, 1 == s, s != "x")
