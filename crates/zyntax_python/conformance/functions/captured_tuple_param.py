def attacks(pos, rows):
    row = list(filter(lambda r: pos in r, rows))
    return len(row), pos


def counted(n):
    for i in range(n):
        pass
    print(i)
    for j in range(0, 10, 3):
        pass
    print(j)
    for k in range(5):
        if k == 2:
            break
    print(k)
    for m in range(10, 0, -4):
        pass
    print(m)
    total = []
    for x in range(3):
        for x in range(2):
            pass
        total.append(x)
    print(total)


def main():
    rows = [[(i, j) for j in range(3)] for i in range(3)]
    for i in range(2):
        for j in range(3):
            print(attacks((i, j), rows))
    counted(4)
    print("abcabc".index("c"), "héllo".index("l"))
    try:
        "abc".index("z")
    except ValueError:
        print("ValueError")


main()
