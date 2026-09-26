def store(o, k, v):
    o[k] = v

def main():
    rec = {"a": 1, "b": 2}
    other = {"a": 3, "b": 4}
    holder = [other, 1]
    store(holder[0], "c", 5)
    print(rec["a"] + rec["b"], other)
    rec["a"] = 10
    print(rec)

main()
