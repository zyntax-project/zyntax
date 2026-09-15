# String keys looked up as strings: stored through a typed string or a
# dynamic value, read back either way, in a small dict and one with a
# table, with missing keys, defaults, membership and deletion.

def dynamic(v):
    return v


counts = {}
words = "alpha beta gamma alpha beta alpha".split(" ")
for w in words:
    counts[w] = counts.get(w, 0) + 1
print(counts)
print(counts["alpha"], counts.get("beta"), counts.get("delta"), counts.get("delta", -1))
print("gamma" in counts, "delta" in counts)

big = {}
for i in range(40):
    big["key" + str(i)] = i
key = dynamic("key17")
print(len(big), big["key17"], big[key], big.get(key), "key39" in big, "key40" in big)
big[key] = 170
print(big["key17"])
big["key17"] = 1700
print(big[key], big.get("key17", 0))
del big["key17"]
print("key17" in big, len(big))
try:
    print(big["key17"])
except KeyError as e:
    print("KeyError", e)

mixed = {1: "one", "1": "string one", 2.5: "float"}
print(mixed["1"], mixed[1], "1" in mixed, 1 in mixed)
mixed["1"] = "changed"
print(mixed)
