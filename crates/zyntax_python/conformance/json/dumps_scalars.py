# json.dumps of every scalar kind, alone and inside containers, from
# values that reach it dynamically
import json


def dump(x):
    return json.dumps(x)


def flags(n):
    out = []
    for i in range(n):
        out.append(dump(i % 2 == 0))
        out.append(dump({"i": i, "even": i % 2 == 0, "half": i / 2, "name": str(i)}))
    return out


for value in [True, False, None, 0, -12, 3.5, -0.0, 1e21, "text", [True, False], (None, 1), {"k": True}]:
    print(dump(value))
print(flags(3))
print(dump([[True], [False, None], {"nested": {"deep": [1, 2.5, "x", True]}}]))
print(dump({"a": 1, "b": [False]}) == '{"a": 1, "b": [false]}')
total = 0
for i in range(1000):
    if dump(i % 3 == 0) == "true":
        total += 1
print(total)
