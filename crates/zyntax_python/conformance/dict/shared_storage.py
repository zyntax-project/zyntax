# A dict or set reached through a list, a loop variable, a field or a
# callee is one dict or set: a write through any of them is seen by all.
class Holder:
    def __init__(self):
        self.m = {}
        self.s = set()

    def fill(self, n):
        for i in range(n):
            self.m['k' + str(i)] = [i] * 2
            self.s.add(i * 7)


def bump(rec):
    rec['w'] = rec['w'] + 10


def main():
    points = []
    for i in range(4):
        points.append({'x': i, 'w': 1})
    for p in points:
        p['w'] = 7
    bump(points[0])
    print(points)
    h = Holder()
    h.fill(5)
    other = h.m
    other['extra'] = [0]
    print(sorted(h.m), sorted(h.s), len(other))
    grid = {}
    for x in range(3):
        grid[(x, x)] = x
    view = [grid]
    view[0][(9, 9)] = 81
    print(grid)


main()
