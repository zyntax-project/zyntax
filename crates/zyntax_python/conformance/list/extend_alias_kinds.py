# += and extend on a list of one kind by values of another kind change
# the one list every alias sees

a = [1, 2]
b = a
a += [1.5]
print(b, a is b)

def f(o):
    same = o
    o += ["s"]
    print(same, o is same)
    o.extend([2.5])
    print(same, o is same)

f([1, 2])
f(["x"])
f([1.0])

ys = [1, 2]
same = ys
ys += (3,)
ys += ('a',)
print(same, ys is same)

def g(v):
    w = v
    v += [None]
    return w
print(g([1]))
print(g([1.5]))
def h():
    c = [1, 2]
    d = c
    c += ["x"]
    c.extend([None])
    print(d, c is d)
h()
