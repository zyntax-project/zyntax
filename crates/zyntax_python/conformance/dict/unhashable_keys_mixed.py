def attempt(label, f):
    try:
        print(label, f())
    except TypeError as e:
        print(label, "TypeError", e)
    except KeyError as e:
        print(label, "KeyError", e)

d = {1: "x", "a": 2, (1, 2): 3}
s = {1, "a", (1, 2)}
attempt("obj dict getitem", lambda: d[[1]])
attempt("obj dict in", lambda: [1] in d)
attempt("obj dict get", lambda: d.get([1]))
attempt("obj dict pop", lambda: d.pop([1]))
attempt("obj dict pop default", lambda: d.pop([1], 0))
attempt("obj dict setdefault", lambda: d.setdefault([1], 0))
attempt("obj set in", lambda: [1] in s)
attempt("obj set add", lambda: s.add([1]))
attempt("obj set remove", lambda: s.remove([1]))
attempt("obj set discard", lambda: s.discard([1]))
attempt("obj dict dict", lambda: d[{}])
attempt("obj set set", lambda: {1} in s)
attempt("frozenset ok", lambda: frozenset({1}) in s)
def store():
    d[[1]] = 3
attempt("obj dict store", store)
def delete():
    del d[[1]]
attempt("obj dict del", delete)
f = {frozenset({1}): 1}
attempt("frozenset key", lambda: f[frozenset({1})])
