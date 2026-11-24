import abc
from abc import ABC, abstractmethod


class Shape(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass

    def describe(self):
        return self.name + " " + str(self.area())


class Square(Shape):
    def __init__(self, side):
        super().__init__("square")
        self.side = side

    def area(self):
        return self.side * self.side

    def perimeter(self):
        return 4 * self.side


class Flat(Shape):
    def area(self):
        return 0


class Animal(abc.ABC):
    @abc.abstractmethod
    def sound(self):
        ...


class Dog(Animal):
    def sound(self):
        return "woof"


class Puppy(Dog):
    pass


print(Square(3).describe(), Square(2).perimeter())
for make in [lambda: Shape("x"), lambda: Flat("flat"), lambda: Animal()]:
    try:
        make()
    except TypeError as e:
        print(e)
print(Dog().sound(), Puppy().sound())
print(isinstance(Square(1), Shape), isinstance(Dog(), Animal))
