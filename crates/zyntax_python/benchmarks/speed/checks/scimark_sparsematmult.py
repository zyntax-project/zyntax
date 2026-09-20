import scimark
from array import array
N = 8
x = array('d', [0]) * N
y = array('d', [0]) * N
for i in range(N):
    x[i] = 1.0 + i * 0.5
val = array('d', [0]) * 16
col = array('i', [0]) * 16
row = array('i', [0]) * (N + 1)
for r in range(N):
    row[r + 1] = row[r] + 2
    col[2 * r] = r
    col[2 * r + 1] = (r + 3) % N
    val[2 * r] = 0.25 * (r + 1)
    val[2 * r + 1] = -0.5
scimark.SparseCompRow_matmult(N, y, val, row, col, x, 3)
print([round(v, 9) for v in y])
print(scimark.SparseMatMult(["10", "50", "2"]))
