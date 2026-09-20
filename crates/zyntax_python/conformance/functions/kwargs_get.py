class Surface(object):
    def __init__(self, **kwargs):
        self.colour = kwargs.get('colour', (1, 1, 1))
        self.shine = kwargs.get('shine', 0.2)
        self.name = kwargs.get('name')


class Checker(Surface):
    def __init__(self, **kwargs):
        Surface.__init__(self, **kwargs)
        self.other = kwargs.get('other', (0, 0, 0))
        self.size = kwargs.get('size', 1)


def describe(**kwargs):
    return "%s/%d" % (kwargs.get('label', 'none'), kwargs.get('count', 0))


def wrap(**kwargs):
    return describe(**kwargs) + "!"


def main():
    a = Surface()
    b = Surface(colour=(0.5, 0.25, 0.0), name="b")
    c = Checker(size=3, shine=0.9)
    print(a.colour, a.shine, a.name)
    print(b.colour, b.shine, b.name)
    print(c.colour, c.shine, c.name, c.other, c.size)
    print(describe(), describe(label="x", count=2), wrap(count=7))


main()
