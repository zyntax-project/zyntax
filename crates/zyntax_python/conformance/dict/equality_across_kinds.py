# Dicts are equal when their keys are equal across kinds and each value
# equals the other's value for that key.
print({1: 2} == {1.0: 2.0}, {1: 2} == {1.0: 3})
print({(1, 2): "a"} == {(1.0, 2.0): "a"}, {"a": 1} == {"a": True})
print({1: 2, 3: 4} == {3: 4, 1: 2}, {1: 2} == {1: 2, 3: 4})
a = {}
b = {}
for i in range(30):
    a[i] = i
    b[float(29 - i)] = float(29 - i)
print(a == b, a != b)
b[0.0] = 1
print(a == b)
print({} == {}, {1: 2} == [1], {1: 2} == {1})
