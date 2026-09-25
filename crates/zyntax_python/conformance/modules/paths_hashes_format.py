import os
import hashlib


class Node(object):
    def __init__(self, code):
        self.code = code
        self.symbol = None

    def __lt__(self, other):
        return self.code < other.code

    def __repr__(self):
        return {}.format(self.code)


class Table(object):
    def __init__(self, codes):
        l = []
        for code in codes:
            l.append(Node(code))
        l.sort()
        self.table = l

    def populate(self):
        for i, x in enumerate(self.table):
            x.symbol = i * 10
            x.reverse_symbol = -i


def main():
    # The separator join inserts is the platform's; the case asks that
    # it inserts one.
    print(os.path.join("a", "b.txt").replace("\\", "/"), os.path.dirname("x/y/z.py"), os.path.basename("x/y/z.py"))
    print(os.path.exists("no_such_file_here"), os.path.join("a/", "b"))
    print(hashlib.md5(b"hello world").hexdigest(), len(hashlib.md5(b"").digest()))
    print(hashlib.md5("héllo".encode("utf-8")).hexdigest(), b"\x01\xab".hex())
    print("{} and {}".format(1, "two"), "{0}-{1}-{0}".format("a", "b"), "{x}:{y:>4}".format(x=1, y=2))
    print("{:.3f} {:>6} {!r} {{literal}}".format(3.14159, "ab", "q"))
    print(hex(255), oct(8), bin(5), hex(-1))
    t = Table([5, 1, 3])
    t.populate()
    print([(n.code, n.symbol, n.reverse_symbol) for n in t.table])
    nodes = [Node(2), Node(1)]
    print(nodes[0] < nodes[1], nodes[1] < nodes[0], max(nodes).code, min(nodes).code)
    print(ord(b"A"), ord(b"\xff"), ord("é"))
    files = [open(__file__, "rb"), open(__file__, "r")]
    print(files[0].read(6), files[1].read(6))
    for f in files:
        f.close()
    print("áé".find("é"), "abcabc".find("c"), "xyz".find("q"))


main()
