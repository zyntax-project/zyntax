# test_dict: keys/values/items
d = {1: "one", 2: "two", 3: "three"}
print(list(d.keys()))
print(list(d.values()))
for k, v in d.items():
    print(k, v)
print({x: x * x for x in range(4)})
