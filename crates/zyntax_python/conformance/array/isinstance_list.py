# An array is not a list: isinstance tells them apart, on a value of
# known type and on one that arrives boxed.
from array import array


def kinds(x):
    return (isinstance(x, list), isinstance(x, array), isinstance(x, tuple))


a = array("d", [1.0])
print(isinstance(a, list), isinstance(a, array))
print(isinstance([1.0], list), isinstance([1.0], array))
print(isinstance(array("i", [1, 2]), list))

for v in [array("d", [1.0]), [1.0], (1, 2), array("i", [1, 2]), array("b", [3])]:
    print(kinds(v))

if isinstance(a, list):
    print("list")
else:
    print("not list")


class C:
    def __init__(self):
        self.d = array("d", [1.0])


for o in (C(), 1):
    x = o.d if o != 1 else array("d", [2.0])
    print(isinstance(x, list), isinstance(x, array))
