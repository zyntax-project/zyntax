x = 3.14159
name = "bob"
age = 7
d = {"a": "z", "b": 7}
print("%(a)s %(b)d" % {"a": "z", "b": 7})
print("%(a)s-%(a)s %(b)05d|%(b)-4d|" % d)
print("%(n)s is %(x).2f" % {"n": name, "x": x})
print("%(pct)d%% done" % {"pct": 50})
print("%*d|%-*d|" % (5, 3, 5, 4))
print("%.*f" % (2, x))
print("%*.*f|" % (9, 3, x))
print("%0*d" % (6, 42))
w = 7
print("%*s|%-*s|" % (w, "ab", w, "cd"))
print("%s %*d" % ("n", 4, 12))
