# isinstance() of a value that is an instance of one of several classes,
# or None.
class A(object):
    pass
class B(A):
    pass
class C(object):
    pass
def pick(i):
    if i == 0:
        return A()
    if i == 1:
        return B()
    if i == 2:
        return C()
    return None
for i in range(4):
    v = pick(i)
    print(i, isinstance(v, A), isinstance(v, B), isinstance(v, C), isinstance(v, object), v is None)
x = A() if pick(0) else None
print(isinstance(x, A), isinstance(None, A))
