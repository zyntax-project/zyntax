class Array2D(object):
    def __init__(self, w, h):
        self.width = w
        self.height = h

    def size(self):
        return self.width * self.height


class ArrayList(Array2D):
    def size(self):
        return -self.width * self.height


def SOR(n):
    return "SOR(%d)" % n


def LU(n):
    return "LU(%d)" % n


def run(names):
    for name in names:
        f = eval(name)
        print(f(3))


def main():
    n, cycles, Array = map(eval, ["4", "2", "Array2D"])
    a = Array(n, n)
    print(n, cycles, a.size())
    Array = eval("ArrayList")
    print(Array(2, 3).size())
    print(eval("1.5") + 1, eval("-7"))
    run(["SOR", "LU"])
    make = Array2D
    print(make(2, 2).size())
    ctors = [Array2D, ArrayList]
    print([c(2, 5).size() for c in ctors])
    try:
        eval("nothing")
    except NameError:
        print("NameError")


main()
