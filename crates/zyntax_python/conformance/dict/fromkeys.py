# dict.fromkeys builds a dict with one value for every key.

print(dict.fromkeys(["a", "b", "c"]))
print(dict.fromkeys(["a", "b", "c"], 0))
print(dict.fromkeys("abca", 1))
print(dict.fromkeys(range(3), "x"))
print(dict.fromkeys([], 5))

keys = [3, 1, 2, 1]
counts = dict.fromkeys(keys, 0)
for k in keys:
    counts[k] += 1
print(counts)

# The one value is shared by every key.
shared = dict.fromkeys(["p", "q"], [])
shared["p"].append(1)
print(shared)

def make():
    print("made")
    return [0]

built = dict.fromkeys(["x", "y"], make())
built["x"].append(9)
print(built)
print(len(dict.fromkeys((1, 2, 2, 3))))
