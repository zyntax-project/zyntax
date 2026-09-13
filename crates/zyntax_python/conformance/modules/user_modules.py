# importing the program's own modules: whole, by name, aliased, dotted
import lib.shapes
from lib.shapes import Circle, perimeter
import lib.text as text
from lib import shapes

print(lib.shapes.area(1), shapes.area(2))
print(round(perimeter(1), 3))
c = Circle(3)
print(c.area(), c.describe())
print(text.banner("hi"), text.circle_line(1))
print(lib.shapes.calls_so_far(), shapes.count, lib.shapes.PI_ISH)
print(__name__)

def local_area(r):
    # a local shadows nothing from the module
    area = r * r
    return area

print(local_area(4))
print([shapes.area(r) for r in [1, 2]])
