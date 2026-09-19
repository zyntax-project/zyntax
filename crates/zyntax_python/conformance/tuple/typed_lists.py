# Lists of tuples of one shape are typed: built, appended, indexed,
# iterated, searched, sorted and printed with no box per element.
def total(points):
    s = 0.0
    for (x, y) in points:
        s += x * y
    return s


pts = [(1.0, 2.0), (3.0, 4.0)]
pts.append((5.0, 6.0))
print(total(pts), len(pts), pts[1], pts[-1][0])
print(pts)
q = [(i, i * i) for i in range(4)]
print(q, q[2][1], (2, 4) in q, (2, 5) in q, q.index((3, 9)))
q.sort()
q.reverse()
print(q)
