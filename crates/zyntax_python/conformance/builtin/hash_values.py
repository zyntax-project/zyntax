def attempt(label, f):
    try:
        print(label, f())
    except TypeError as e:
        print(label, "TypeError", e)

class P:
    def __init__(self, x):
        self.x = x
    def __hash__(self):
        return self.x

class Q:
    def __eq__(self, other):
        return True

def make(flag):
    if flag:
        return [1]
    return "k"

attempt("list", lambda: hash([1]))
attempt("dict", lambda: hash({}))
attempt("set", lambda: hash({1}))
attempt("frozenset", lambda: hash(frozenset({1})) == hash(frozenset({1})))
attempt("dynamic list", lambda: hash(make(True)))
attempt("dynamic str", lambda: hash(make(False)) == hash("k"))
attempt("class hash", lambda: hash(P(7)))
attempt("class eq only", lambda: hash(Q()))
attempt("tuple of list", lambda: hash((1, [2])))
attempt("float", lambda: hash(2.0))
attempt("equal values", lambda: hash(3) == hash(3.0) == hash(True + 2))
x = 5
attempt("name", lambda: hash(x))
