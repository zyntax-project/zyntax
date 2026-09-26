def add_c(m):
    m["c"] = 7

def main():
    d = {"k": 1}
    d["b"] = 2
    s = {"k": 3}
    print(len(s), s)
    add_c(s)
    print(s)
    t = {"x": 1, "y": 2}
    key = "y"
    print(t[key])
    u = {"p": 1, "q": 2}
    del u["p"]
    print(u)
    v = {"m": 1, "n": 2}
    v.update({"o": 3})
    print(v, v.pop("m"), v)
    w = {"r": 1}
    try:
        print(w["nope"])
    except KeyError as e:
        print("KeyError", e)
    z = {"r": 5}
    print(z["r"] + 1)

main()
