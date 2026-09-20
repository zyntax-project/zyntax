import scimark
rnd = scimark.Random(7)
a = rnd.RandomMatrix(scimark.Array2D(6, 6))
scimark.SOR_execute(1.25, a, 4)
print([round(v, 9) for v in a.data])
b = rnd.RandomMatrix(scimark.ArrayList(6, 6))
scimark.SOR_execute(1.25, b, 4)
print([[round(v, 9) for v in row] for row in b.data])
print(scimark.SOR(["10", "3", "Array2D"]), scimark.SOR(["10", "3", "ArrayList"]))
