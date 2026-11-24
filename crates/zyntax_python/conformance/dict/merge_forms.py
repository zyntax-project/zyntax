defaults = {"color": "red", "size": "m"}
custom = {"size": "xl"}
merged = {**defaults, **custom, "extra": "yes"}
print(sorted(merged.items()))
print(sorted({**custom}.items()))
print(sorted({"first": 1, **{"second": 2}, "third": 3}.items()))

counts = {"a": 1}
counts.update(b=2, c=3)
print(sorted(counts.items()))
counts.update({"z": 26}, y=25)
print(sorted(counts.items()))
counts.update([("p", 16), ("q", 17)])
print(sorted(counts.items()))
counts.update((k, len(k)) for k in ["alpha", "be"])
print(sorted(counts.items()))

built = dict(x=1.5, y=2.5)
print(sorted(built.items()))
print(sorted(dict([("k", 1)], j=2).items()))
print(sorted(dict(built, z=3.5).items()))
extra = {"w": 4.5}
print(sorted(dict(built, **extra).items()))
print(len(dict(**built)))

def options(base, **overrides):
    return {**base, "origin": "call"}

def settings(host="h", port=80):
    cfg = {"host": host}
    cfg.update(port=port, debug=False)
    return cfg
print(sorted(settings().items()))
print(sorted(settings("x", 8).items()))

class Registry:
    def __init__(self):
        self.entries = {}
    def add_all(self, **named):
        self.entries.update(a=1, b=2)
        return len(self.entries)
print(Registry().add_all())

def copy_of(x):
    return dict(x)
print(sorted(copy_of({"m": 1}).items()))
print(sorted(copy_of([("n", 2)]).items()))
print(sorted(copy_of({"m": 1}, ).items()) == sorted(copy_of({"m": 1}).items()))
