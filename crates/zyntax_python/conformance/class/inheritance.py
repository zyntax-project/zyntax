# single inheritance, method override, super()
class Animal:
    def __init__(self, name: str):
        self.name = name
    def speak(self) -> str:
        return "..."
    def describe(self) -> str:
        return self.name + " says " + self.speak()

class Dog(Animal):
    def speak(self) -> str:
        return "woof"

class Puppy(Dog):
    def speak(self) -> str:
        return super().speak() + "!"

print(Animal("thing").describe())
print(Dog("rex").describe())
print(Puppy("bit").describe())
