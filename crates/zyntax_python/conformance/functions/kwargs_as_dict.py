def show(*args, **opts):
    return len(args), sorted(opts.items())
print(show(1, 2, x=3, y=4))
print(show())
print(show(7, **{"a": 1}))
d = {"p": 1, "q": 2}
print(show(0, **d))
print(show(*[1, 2, 3], z=0, **d))

def build(name, size=1, **extra):
    extra["name"] = name
    extra["size"] = size
    return extra
print(sorted(build("a").items()), sorted(build("b", 2, color="red").items()))
print(sorted(build(name="c", size=5, shape="sq").items()))

def opts(**kw):
    out = []
    for k in sorted(kw):
        out.append(k + "=" + str(kw[k]))
    return ", ".join(out), len(kw), "x" in kw, kw.get("x", "none")
print(opts(), opts(x=1, y=2))

def forward(**kw):
    return opts(**kw)
print(forward(a=1))

class Widget:
    def __init__(self, label, **attrs):
        self.label = label
        self.attrs = attrs
    def describe(self, **more):
        merged = dict(self.attrs)
        merged.update(more)
        return self.label, sorted(merged.items())
w = Widget("w", color="red", size=3)
print(w.attrs, w.describe(size=9, extra=True))

def wrapper(fn_name, *args, **kwargs):
    return fn_name, args, kwargs
print(wrapper("f", 1, 2, k=3))


def f(a, *args, b=2, **kw):
    return a, args, b, kw


print(f(1, 2, 3, b=4, c=5))
print(f(*[1, 2], **{"b": 7}))
print(f(1), f(1, b=0), f(1, x=1, y=2))
