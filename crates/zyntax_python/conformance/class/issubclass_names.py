class Animal:
    pass
class Dog(Animal):
    pass
class Puppy(Dog):
    pass
class Rock:
    pass
print(issubclass(Dog, Animal), issubclass(Animal, Dog), issubclass(Puppy, Animal))
print(issubclass(Dog, Dog), issubclass(Rock, Animal), issubclass(Puppy, Rock))
print(issubclass(Puppy, (Rock, Dog)), issubclass(Rock, (Animal, Dog)))
print(issubclass(Dog, object), issubclass(bool, int), issubclass(int, bool), issubclass(str, str))
print(issubclass(Dog, int), issubclass(int, Dog))
