# Values that are an instance of one of several unrelated classes: fields at
# one offset or not, methods some classes lack, None, tuples and lists of them.
class Vec(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def kind(self):
        return "vec"
    def scale(self, k):
        return Vec(self.x * k, self.y * k)
    def __repr__(self):
        return "Vec(%r, %r)" % (self.x, self.y)

class Pt(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def kind(self):
        return "pt"
    def scale(self, k):
        return Pt(self.x * k, self.y * k)
    def __repr__(self):
        return "Pt(%r, %r)" % (self.x, self.y)

class Odd(object):
    def __init__(self, n):
        self.label = "odd"
        self.x = n
    def kind(self):
        return 7
    def scale(self, k):
        return self

def sub(a, b):
    if isinstance(b, Pt):
        return Vec(a.x - b.x, a.y - b.y)
    return Pt(a.x - b.x, a.y - b.y)

def pick(i):
    if i == 0:
        return Vec(1, 2)
    if i == 1:
        return Pt(3.5, 4)
    return None

r = sub(Pt(5, 5), Pt(1, 2))
q = sub(Pt(5, 5), Vec(1, 2))
print(r, q, r.x + q.y, r.kind(), q.kind())
for i in range(3):
    v = pick(i)
    if v is None:
        print("none", v is None, bool(v))
    else:
        print(v.kind(), v.x, v.scale(2), bool(v), str(v))
v = pick(1)
v.x = 9
print(v, v.x)
things = [pick(0), pick(1), Odd(4)]
for t in things:
    print(t.kind(), t.x)
w = Odd(3) if r.x > 0 else pick(0)
print(w.kind(), w.x, w.scale(1).x)

class A(object):
    def __init__(self, n):
        self.n = n
    def hit(self):
        return self.n
class B(object):
    def __init__(self, n):
        self.n = n * 10
    def hit(self):
        return self.n + 1
class S(object):
    pass
class Scene(object):
    def __init__(self):
        self.objects = []
    def add(self, o, s):
        self.objects.append((o, s))
sc = Scene()
sc.add(A(1), S())
sc.add(B(2), S())
for o, s in sc.objects:
    print(o.hit())
pairs = [(A(3), 1), (B(4), 2)]
for o, k in pairs:
    print(o.hit() + k)
