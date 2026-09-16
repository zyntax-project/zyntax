def unpack(value):
    first, second = value
    print(first, second)


unpack((1, 2))
unpack([3, 4])
unpack("xy")
unpack({5: "five", 6: "six"})


def nested(value):
    (first, second), third = value
    print(first, second, third)


nested(((7, 8), 9))

for value in ([], [1], [1, 2, 3]):
    try:
        unpack(value)
    except ValueError:
        print("value error")

alias = [[], 2]
(alias[:], last) = alias
print(alias, last)
