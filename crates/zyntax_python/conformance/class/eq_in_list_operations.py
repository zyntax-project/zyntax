class P:
    def __init__(self, v):
        self.v = v

    def __eq__(self, o):
        return isinstance(o, P) and self.v == o.v

    def __repr__(self):
        return "P(" + str(self.v) + ")"


class Q(P):
    pass


a = [P(1), P(2), P(3)]
print(P(2) in a, P(9) in a, P(2) not in a, Q(2) in a)
print(a.index(P(3)), a.count(P(1)), a.count(P(5)))
a.remove(P(2))
print(a, len(a))
print([P(1)] == [P(1)], [P(1)] == [P(2)], [P(1), P(2)] != [P(1), P(2)], (P(1), 2) == (P(1), 2), P(1) != P(1))
b = a + [P(7)]
print(b.index(P(7)), P(7) in b, b.count(Q(7)))
try:
    a.index(P(99))
except ValueError:
    print("ValueError")
seen = []
for p in [P(1), P(1), P(2), P(1)]:
    if p not in seen:
        seen.append(p)
print(seen)
