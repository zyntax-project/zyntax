# a module of the program: functions, a class, a module variable, and its own import
import math

PI_ISH = 3
count = 0

def area(r):
    global count
    count += 1
    return round(math.pi * r * r, 2)

def perimeter(r: float) -> float:
    return 2 * math.pi * r

class Circle:
    def __init__(self, r):
        self.r = r

    def area(self):
        return area(self.r)

    def describe(self):
        return f"circle r={self.r} in {__name__}"

def calls_so_far():
    return count

print("shapes loaded as", __name__)
