import scimark
from array import array
rnd = scimark.Random(7)
A = rnd.RandomMatrix(scimark.ArrayList(6, 6))
lu = scimark.ArrayList(6, 6)
lu.copy_data_from(A)
pivot = array('i', [0]) * 6
scimark.LU_factor(lu, pivot)
print([[round(v, 9) for v in row] for row in lu.data], list(pivot))
print(scimark.LU(["5", "2"]))
