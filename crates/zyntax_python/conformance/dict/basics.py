# test_dict: literal, get/set, in, len, iteration order
d = {"a": 1, "b": 2}
print(d["a"])
d["c"] = 3
print(len(d))
print("b" in d, "z" in d)
for k in d:
    print(k, d[k])
print(d.get("a"))
print(d.get("z", 0))
del d["a"]
print(len(d))
print({})
