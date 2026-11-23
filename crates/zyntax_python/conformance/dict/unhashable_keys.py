def attempt(label, f):
    try:
        print(label, f())
    except TypeError as e:
        print(label, "TypeError", e)
    except KeyError as e:
        print(label, "KeyError", e)

s = {1, 2, 3}
t = {"a", "b"}
d = {1: "x", 2: "y"}
e = {"a": 1}
big = set(range(20))
bigd = {i: i for i in range(20)}
attempt("set in int", lambda: [1] in s)
attempt("set in str", lambda: [1] in t)
attempt("set in big", lambda: [1] in big)
attempt("set not in", lambda: [1] not in s)
attempt("dict getitem", lambda: d[[1]])
attempt("dict in", lambda: [1] in d)
attempt("dict in str", lambda: [1] in e)
attempt("dict get", lambda: d.get([1]))
attempt("dict big getitem", lambda: bigd[[1]])
attempt("dict big in", lambda: [1] in bigd)
attempt("dict dict key", lambda: {1: 2}[{}])
attempt("dict set key", lambda: d[{1}])
attempt("set add", lambda: s.add([1]))
attempt("set remove", lambda: s.remove([1]))
attempt("set discard", lambda: s.discard([1]))
attempt("dict pop", lambda: d.pop([1], 0))
