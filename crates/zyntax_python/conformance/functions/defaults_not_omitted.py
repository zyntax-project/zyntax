# A default is one of the values a parameter takes at each call that
# leaves it out; a call that passes every argument contributes only
# what it passes.
LIMITS = [1.5, 2.5]


def scale(xs, factor=2.5):
    return [x * factor for x in xs]


def clamp(v, lo=0, hi=10):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def bounded(v, limits=LIMITS):
    return limits[0] <= v <= limits[1]


def greet(name, greeting="hi", punct="!"):
    return greeting + " " + name + punct


print(scale([1, 2], 3), scale([1, 2], 0.5))
print(clamp(-1, -5, 5), clamp(11, 0, 20), clamp(7, 3, 5))
print(bounded(2.0, [1, 3]), bounded(4.0, [1, 3]))
print(greet("a", "yo", "?"), greet("b", punct="."), greet("c"))
print(scale([3], 2) + scale([1.5]))
