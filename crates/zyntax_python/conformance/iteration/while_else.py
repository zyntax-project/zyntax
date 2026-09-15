def ended_normally():
    x = 0
    while x < 2:
        x += 1
    else:
        return x + 10


def broke():
    while True:
        break
    else:
        return 99
    return 1


def continued():
    x = 0
    while x < 2:
        x += 1
        continue
    else:
        return x + 20


def nested_break():
    x = 0
    while x < 2:
        while True:
            break
        x += 1
    else:
        return x + 30


def finally_break():
    while True:
        try:
            break
        finally:
            pass
    else:
        return 99
    return 2


def finally_continue():
    x = 0
    while x < 2:
        try:
            break
        finally:
            x += 1
            continue
    else:
        return x + 40


def exception():
    try:
        while True:
            raise ValueError("oops")
        else:
            return 99
    except ValueError:
        return 3


print(ended_normally())
print(broke())
print(continued())
print(nested_break())
print(finally_break())
print(finally_continue())
print(exception())

x = 0
while x:
    print("wrong")
else:
    print("zero")
