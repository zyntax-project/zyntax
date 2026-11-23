# int() and float() of a string held as a dynamic value raise ValueError.
rows = [["3", "x"]]
v = rows[0][1]
try:
    print(int(v))
except ValueError as e:
    print("ValueError", e)
try:
    print([int(s) for s in rows[0]])
except ValueError as e:
    print("ValueError", e)
try:
    print(float(v))
except ValueError as e:
    print("ValueError", e)
