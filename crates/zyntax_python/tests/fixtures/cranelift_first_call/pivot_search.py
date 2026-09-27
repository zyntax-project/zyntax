from array import array


class Rows:
    def __init__(self, n):
        self.n = n
        self.data = [array('d', [0]) * n for y in range(n)]

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self.data[idx[1]][idx[0]]
        else:
            return self.data[idx]


def fill(g, seed):
    x = seed
    for i in range(g.n):
        for j in range(g.n):
            x = (x * 1103515245 + 12345) % 2147483648
            g.data[i][j] = x / 2147483648.0 - 0.5


def pivots(A, piv):
    n = A.n
    for j in range(n):
        jp = j
        t = abs(A[j][j])
        for i in range(j + 1, n):
            ab = abs(A[i][j])
            if ab > t:
                jp = i
                t = ab
        piv[j] = jp


g = Rows(7)
fill(g, 7)
other = Rows(1)
other.data = 5
piv = [0] * 7
pivots(g, piv)
print(piv)
