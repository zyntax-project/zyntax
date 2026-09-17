values = [1.0, 2.0]
print(values[-1])
values[-1] = 3.0
print(values)

for index in (2, -3):
    try:
        print(values[index])
    except IndexError:
        print("read out of range")
    try:
        values[index] = 4.0
    except IndexError:
        print("write out of range")

print(values)
