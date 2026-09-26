# A return, break, continue or raise leaving an except or else clause
# runs finally first.
def f(i: int) -> int:
    try:
        if i == 0:
            raise ValueError("zero")
        return i
    except ValueError as e:
        print("handler", e)
        return -1
    finally:
        print("finally", i)


print(f(0))
print(f(5))


def g() -> int:
    total = 0
    for i in range(5):
        try:
            if i % 2 == 0:
                raise KeyError("even")
            total += 10
        except KeyError:
            if i == 4:
                break
            continue
        finally:
            print("finally", i)
            total += 1
    return total


print(g())


def h(x: int) -> str:
    try:
        n = x + 1
    except ValueError:
        return "handler"
    else:
        if n > 3:
            return "else " + str(n)
    finally:
        print("finally h", x)
    return "end"


print(h(1))
print(h(5))


def k() -> int:
    try:
        raise ValueError("first")
    except ValueError:
        raise KeyError("second")
    finally:
        print("finally k")


try:
    k()
except KeyError as e:
    print("KeyError", e)
