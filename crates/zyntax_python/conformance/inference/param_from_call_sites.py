# an unannotated parameter takes the type of what every call passes
def twice(x):
    return x + x

def show(label, value):
    print(label, value)

print(twice(3), twice(2.5))
show("int", twice(4))
show("str", "ab" + "c")
show(value=1.5, label="kw")

def total(xs, start=0):
    for x in xs:
        start += x
    return start

print(total([1, 2, 3]), total([1.5, 2.5], 1))
print(total([1, 2], start=10))

def only_none(x=None):
    return x is None

print(only_none(), only_none(3), only_none(None))

def flag(b):
    return "yes" if b else "no"

print(flag(True), flag(False), flag(0))
