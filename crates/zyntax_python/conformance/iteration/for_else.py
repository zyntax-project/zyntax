def counted():
    for i in range(3):
        if i == 5:
            break
    else:
        return 10
    return 0

def counted_break():
    for i in range(3):
        if i == 1:
            break
    else:
        return -1
    return i

def over_list(xs):
    for x in xs:
        if x == 0:
            break
    else:
        return "none"
    return "found"

def over_gen():
    def g():
        yield 1
        yield 2
    for v in g():
        if v == 3:
            break
    else:
        return 5
    return -1

def nested_for():
    total = 0
    for i in range(2):
        for j in range(3):
            if j == 2:
                break
        else:
            total += 100
        total += j
    else:
        total += 1
    return total

def in_try():
    try:
        for i in range(3):
            try:
                if i == 1:
                    break
            finally:
                pass
        else:
            return -1
        return i
    except ValueError:
        return -2

def over_dict(d):
    n = 0
    for k in d:
        n += d[k]
    else:
        n += 1000
    return n

def over_str(s):
    for ch in s:
        if ch == "z":
            break
    else:
        return "no z"
    return "z at some point"

print(counted(), counted_break(), over_list([1, 2]), over_list([1, 0]), over_gen())
print(nested_for(), in_try(), over_dict({"a": 1, "b": 2}), over_str("abc"), over_str("xyz"))
for k in range(2):
    for m in range(k):
        break
    else:
        print("else", k)
