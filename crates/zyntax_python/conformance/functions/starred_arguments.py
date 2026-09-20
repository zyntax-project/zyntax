class Canvas(object):
    def __init__(self, w):
        self.w = w
        self.plotted = []

    def plot(self, x, y, r, g, b):
        self.plotted.append((x, y, r, g, b))


class Sink(object):
    def plot(self, x, y, r, g, b):
        print("sink", x, y, r, g, b)


def colour_of(k):
    if k > 1:
        return (1.0, 0.5, 0.0)
    return (0, 0, 1)


def add3(a, b, c):
    return a + b + c


def main():
    Board = Canvas
    c = Board(3)
    colour = (0.1, 0.2, 0.3)
    c.plot(1, 2, *colour)
    dyn = colour_of(2)
    c.plot(3, 4, *dyn)
    print(c.plotted)
    print(add3(*colour), add3(1, *(2, 3)))
    sinks = [Sink(), Canvas(1)]
    for s in sinks:
        s.plot(0, 0, *colour_of(0))
    print(sinks[1].plotted)


main()
