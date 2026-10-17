# A call spreading a tuple whose type is decided after the call is read.
class Canvas(object):
    def __init__(self):
        self.out = []

    def plot(self, x, y, r, g, b):
        self.out.append((x, y, r + g + b))


class Scene(object):
    def colour(self, x):
        if x == 0:
            return (0.0, 0.0, 0.0)
        return self.shade(x)

    def shade(self, x):
        return (x * 0.5, x * 0.25, 1.0)

    def render(self, canvas):
        for x in range(3):
            c = self.colour(x)
            canvas.plot(x, 0, *c)


canvas = Canvas()
Scene().render(canvas)
print(canvas.out)
