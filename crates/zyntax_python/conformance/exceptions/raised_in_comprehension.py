# An element that raises ends the comprehension, and nothing reads the
# list before the exception reaches its handler.
try:
    a, b = [int(s) for s in ["1", "z"]]
    print("unpacked", a, b)
except ValueError as e:
    print("ValueError:", e)

try:
    r = [int(s) for s in ["x"]]
    print(r)
except ValueError as e:
    print("ValueError:", e)


def parse(items):
    out = [int(s) for s in items]
    print("parsed", out)
    return out


try:
    parse(["7", "8"])
    parse(["7", "eight"])
    print("not reached")
except ValueError as e:
    print("ValueError:", e)

i = 0
while i < 2:
    try:
        print([int(s) for s in ["1", "q"][: i + 1]])
    except ValueError as e:
        print("ValueError:", e)
    i += 1

print("last")
a, b = [int(s) for s in ["1", "z"]]
print("not reached")
