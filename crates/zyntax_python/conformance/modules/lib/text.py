# a module importing another module of the program
from lib.shapes import area as circle_area, PI_ISH

def banner(word):
    return "*" * PI_ISH + " " + word.upper() + " " + "*" * PI_ISH

def circle_line(r):
    return f"area {circle_area(r)}"
