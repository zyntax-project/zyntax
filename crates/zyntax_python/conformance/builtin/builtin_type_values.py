print(int(), float(), repr(str()), bool(), list(), dict(), set(), tuple())
factories = [list, int, dict, set, float, str, bool, tuple]
print([repr(make()) for make in factories])
pairs = [(list, [3, 1, 2]), (tuple, "ab"), (set, [1, 1, 2]), (int, "42"), (str, 7), (float, "2.5"), (bool, 0), (dict, [("a", 1)])]
for make, value in pairs:
    print(repr(make(value)))


def build(factory, count):
    out = []
    for _ in range(count):
        out.append(factory())
    return out


print(build(list, 2), build(int, 2), build(dict, 1))
copy = list
original = [1, 2, 3]
duplicate = copy(original)
duplicate.append(4)
print(original, duplicate)


class Holder:
    def __init__(self, factory):
        self.factory = factory

    def make(self):
        factory = self.factory
        return factory()


print(Holder(list).make(), Holder(int).make(), Holder(str).make())
