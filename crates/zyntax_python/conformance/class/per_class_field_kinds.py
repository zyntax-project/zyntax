# A base class's field holds one array; a subclass holds a list of
# arrays in the same field. Methods each class defines read their own
# kind; a method only the base defines reads either, by the instance.
from array import array


class Flat(object):
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.data = array('d', [0]) * (w * h)

    def __getitem__(self, xy):
        x, y = xy
        return self.data[y * self.width + x]

    def __setitem__(self, xy, val):
        x, y = xy
        self.data[y * self.width + x] = val

    def size(self):
        return len(self.data)

    def copy_data_from(self, other):
        self.data[:] = other.data[:]


class Rows(Flat):
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.data = [array('d', [0]) * w for y in range(h)]

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self.data[idx[1]][idx[0]]
        return self.data[idx]

    def __setitem__(self, idx, val):
        if isinstance(idx, tuple):
            self.data[idx[1]][idx[0]] = val
        else:
            self.data[idx] = val

    def copy_data_from(self, other):
        for l1, l2 in zip(self.data, other.data):
            l1[:] = l2


def factor(A):
    n = A.height
    for j in range(n):
        jp = j
        t = abs(A[j][j])
        for i in range(j + 1, n):
            if abs(A[i][j]) > t:
                jp = i
                t = abs(A[i][j])
        if jp != j:
            A[j], A[jp] = A[jp], A[j]
        for k in range(j + 1, n):
            A[k][j] /= A[j][j]
            for jj in range(j + 1, n):
                A[k][jj] -= A[k][j] * A[j][jj]


def main():
    f = Flat(3, 3)
    r = Rows(3, 3)
    for y in range(3):
        for x in range(3):
            f[x, y] = float((x + 1) * (y + 2) % 5 + x)
            r[x, y] = float((x + 2) * (y + 1) % 7 + y)
    g = Flat(3, 3)
    g.copy_data_from(f)
    s = Rows(3, 3)
    s.copy_data_from(r)
    factor(s)
    print(list(g.data), [g[x, 1] for x in range(3)])
    print([[round(v, 6) for v in row] for row in s.data])
    print(f.size(), r.size(), s[1][2] == s[1, 2])
    row = r[0]
    row[0] = 9.5
    print(r[0, 0], r.data[0] is row)
    for obj in (f, r):
        print(len(obj.data), obj.size())


main()
