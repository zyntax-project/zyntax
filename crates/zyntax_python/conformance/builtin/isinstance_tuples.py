class Animal:
    pass


class Dog(Animal):
    pass


class Cat(Animal):
    pass


class Rock:
    pass


def kind(x):
    if isinstance(x, (int, float)):
        return "number"
    if isinstance(x, (str, bytes)):
        return "text"
    if isinstance(x, (list, tuple)):
        return "sequence"
    if isinstance(x, (set, dict)):
        return "collection"
    return "other"


def pet(x):
    return isinstance(x, (Dog, Cat))


def mixed(x):
    return isinstance(x, (Rock, str))


values = [1, 2.5, True, "s", b"b", [1], (1, 2), {1}, {1: 2}, None]
for v in values:
    print(kind(v))

print(pet(Dog()), pet(Cat()), pet(Animal()), pet(Rock()), pet(3))
print(mixed(Rock()), mixed("x"), mixed(1), mixed(Dog()))

n = 3
print(isinstance(n, (int, str)), isinstance(n, (str, bytes)))
print(isinstance(n, (str, (float, (int,)))))
print(isinstance(n, ()))
print(isinstance("a", (list, str)), isinstance([1], (list, str)))
print([isinstance(v, (int, str)) for v in values])
print(sum(1 for v in values if isinstance(v, (list, tuple, set, dict))))
d = Dog()
print(isinstance(d, (Animal,)), isinstance(d, (Cat, Rock)))
