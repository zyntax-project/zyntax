x = 3.14159
w = 8
p = 3
n = 255
s = "hi"
print(f"{x:>{w}.2f}|")
print(f"{x:{w}.{p}}|")
print(f"{x:{w}.{3}f}|{n:{w}}|{s:{w}}|{s:<{w}}|{s:*^{w}}|")
print(f"{n:0{w}d}|{n:0{w}x}|{n:#0{w}x}|{n:+{w}d}|")
print(f"{x:.{p}f}|{x:.{p + 1}e}|{s:.{1}}|")
print("{:{}.{}f}".format(x, 9, 2))
print("{0:>{1}}".format("a", 4))
print("{:>{w}}|{:<{w}}|".format("r", "l", w=5))
print("{val:{width}.{prec}f}".format(val=x, width=10, prec=1))
print("{:{}}".format(n, 6))
for width in (3, 6):
    print(f"[{n:>{width}}]", "[{:>{}}]".format(n, width))
rows = [("ab", 1.5), ("cde", 22.25)]
col = max(len(r[0]) for r in rows)
for name, val in rows:
    print(f"{name:<{col}} {val:>8.2f}")
