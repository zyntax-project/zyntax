class Temp:
    def __init__(self, celsius):
        self._c = 0.0
        self.celsius = celsius

    @property
    def celsius(self):
        return self._c

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("too cold")
        self._c = value

    @property
    def fahrenheit(self):
        return self._c * 9 / 5 + 32


class Circle:
    def __init__(self, r):
        self.r = r

    @property
    def area(self):
        return 3 * self.r * self.r

    @property
    def label(self):
        return "circle(" + str(self.r) + ")"


class Sq:
    def __init__(self, s):
        self.s = s

    @property
    def area(self):
        return self.s * self.s


class Base:
    def __init__(self):
        self._n = 1

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, v):
        self._n = v


class Derived(Base):
    @property
    def n(self):
        return self._n * 10

    @n.setter
    def n(self, v):
        self._n = v + 1


t = Temp(25.0)
print(t.celsius, t.fahrenheit)
t.celsius = 100
print(t.celsius, t.fahrenheit)
t.celsius += 5
print(t.celsius)
try:
    t.celsius = -300
except ValueError as e:
    print("ValueError", e)
print(Circle(2).area, Circle(3).label, Sq(4).area)
shapes = [Circle(1), Sq(2)]
print([s.area for s in shapes], sum(s.area for s in shapes))
b = Base()
d = Derived()
b.n = 5
d.n = 5
print(b.n, d.n)
d.n += 1
print(d.n)
print(t.celsius if t.celsius > 0 else -1)
