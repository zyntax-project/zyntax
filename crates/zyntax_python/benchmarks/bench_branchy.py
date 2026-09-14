# Twenty million iterations through a seven-way branch on the running
# value, reduced modulo a prime each time: branch prediction and
# integer arithmetic.
# Returns 140.

def main() -> int:
    acc = 1
    for i in range(20000000):
        k = acc % 7
        if k == 0:
            acc = acc + 3
        elif k == 1:
            acc = acc * 2 + 1
        elif k == 2:
            acc = acc - 5
        elif k == 3:
            acc = acc + 11
        elif k == 4:
            acc = acc * 3
        elif k == 5:
            acc = acc + 7
        else:
            acc = acc - 1
        acc = acc % 1000003
    return acc

print(main())
