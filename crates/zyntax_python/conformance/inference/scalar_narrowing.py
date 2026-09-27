# A number or None: tests that exclude None before arithmetic and
# comparisons, and what comparing None raises.

def f(n):
    if n % 3 == 0:
        return None
    return n * 0.5

def main():
    tot = 0.0
    cnt = 0
    for i in range(20):
        t = f(i)
        if t is not None and t > 1.0:
            tot += t
        if t is None or t < 2.0:
            cnt += 1
    print(tot, cnt)
    t = f(3)
    try:
        print(t < 1)
    except TypeError as e:
        print("TypeError:", e)
    u = f(4)
    if u is not None:
        print(u + 1.0)

main()
