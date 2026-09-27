import scimark
from array import array

# Another instance holding an int makes ArrayList.data dynamic, so the
# factorisation reads boxed rows.
other = scimark.ArrayList(1, 1)
other.data = 5
rnd = scimark.Random(7)
A = rnd.RandomMatrix(scimark.ArrayList(7, 7))
lu = scimark.ArrayList(7, 7)
lu.copy_data_from(A)
pivot = array('i', [0]) * 7
scimark.LU_factor(lu, pivot)
s = 0.0
for row in lu.data:
    for v in row:
        s += v
print(round(s, 6), list(pivot))
