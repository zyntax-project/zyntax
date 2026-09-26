import json

def make(i):
    return {"id": i, "name": "n" + str(i), "ok": i % 2 == 0}

def main():
    a = make(1)
    b = make(1)
    c = {"id": 1, "name": "n1", "ok": False}
    print(a)
    print(repr(b), str(c))
    print(a == b, a == c, a != c)
    print(len(a), "id" in a, "zz" in a)
    for k in a:
        print(k, a[k])
    print(list(a.keys()), list(a.values()))
    print(a == {"ok": False, "name": "n1", "id": 1})
    boxed = [a, 3, "x"]
    print(boxed)
    print(json.dumps(a))
    print(json.dumps([make(2), make(3)]))
    print(isinstance(a, dict))
    rows = [make(i) for i in range(4)]
    total = 0
    for r in rows:
        if r["ok"]:
            total += r["id"]
        r["id"] = r["id"] * 10
    print(total, [r["id"] for r in rows])
    print(a.get("name"), a.get("missing", 7))
    print(sorted(a.items()))

main()
