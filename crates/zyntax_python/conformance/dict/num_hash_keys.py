# 1, 1.0 and True are one key; a number local probes like any value.


def key(kind):
    k = 1
    if kind == 1:
        k = 1.0
    elif kind == 2:
        k = True
    return k


for insert in range(3):
    d = {}
    k = 1
    if insert == 1:
        k = 1.0
    elif insert == 2:
        k = True
    d[k] = "v%d" % insert
    for probe in range(3):
        p = 1
        if probe == 1:
            p = 1.0
        elif probe == 2:
            p = True
        print(insert, probe, p in d, d.get(p), d[p])
    print(d)
