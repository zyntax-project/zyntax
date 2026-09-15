values = [1, 2]
put = values.insert
take = values.pop
put(1, 9)
print(values, take(0), values, take(), values)

put = values.append
other = values
values = [8]
put(3)
print(values, other)

copy = other.copy
copy_of_other = copy()
copy_of_other.append(7)
print(copy_of_other, other)

remove = other.remove
remove(3)
print(other)

try:
    take(0, 1)
except TypeError:
    print("pop arity")
try:
    take("bad")
except TypeError:
    print("pop index")

class Counter:
    def __init__(self):
        self.value = 0

    def add(self, n):
        self.value += n
        return self.value

    def reset(self):
        self.value = 0

    def add_checked(self, n: int) -> int:
        self.value += n
        return self.value

counter = Counter()
add = counter.add
print(add(3), counter.value)

def apply(f):
    return f(4)

print(apply(add), counter.value)
stored = [add]
print(stored[0](2), counter.value)
reset = counter.reset
print(reset(), counter.value)
checked = counter.add_checked
try:
    checked("bad")
except TypeError:
    print("checked argument", counter.value)
print(checked(5), counter.value)

class DoubleCounter(Counter):
    def add(self, n):
        return super().add(n * 2)

child = DoubleCounter()
bound = child.add
print(bound(3), child.value)

def maybe(flag: bool):
    if flag:
        return Counter()
    return None

try:
    missing = maybe(False).add
except AttributeError:
    print("missing receiver")
