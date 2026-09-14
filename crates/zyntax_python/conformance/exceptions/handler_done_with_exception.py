# A handler that only reads its exception is done with it when it
# leaves, whichever way it leaves: falling through, returning, breaking,
# continuing, raising, or through a nested try.

class Odd(Exception):
    pass

def get(i: int) -> int:
    if i % 2 == 0:
        return i
    raise Odd("odd " + str(i))

def fall_through(n: int) -> int:
    total = 0
    for i in range(n):
        try:
            total += get(i)
        except Odd as e:
            total += len(str(e))
    return total

def early_return(i: int) -> str:
    try:
        return str(get(i))
    except Odd as e:
        return "caught " + str(e)

def breaks(n: int) -> int:
    seen = 0
    for i in range(n):
        try:
            seen += get(i)
        except Odd as e:
            print(f"stop at {e}")
            break
    return seen

def continues(n: int) -> int:
    seen = 0
    for i in range(n):
        try:
            seen += get(i)
        except Odd as e:
            if i > 4:
                print("late " + str(e))
                continue
            seen += 100
    return seen

def nested(i: int) -> str:
    try:
        return str(get(i))
    except Odd as e:
        try:
            return str(get(i + 1)) + " after " + str(e)
        except Odd as inner:
            return "both " + str(e) + " and " + str(inner)

def raises_another(i: int) -> str:
    try:
        return str(get(i))
    except Odd as e:
        raise ValueError("replaced " + str(e))

def reraises(i: int) -> str:
    try:
        return str(get(i))
    except Odd as e:
        if i == 3:
            raise
        return "swallowed " + str(e)

def kept(i: int):
    try:
        return get(i)
    except Odd as e:
        return e

print(fall_through(10))
print(early_return(2), early_return(3))
print(breaks(10))
print(continues(10))
print(nested(1), nested(2))
try:
    raises_another(5)
except ValueError as v:
    print(v)
print(reraises(1))
try:
    reraises(3)
except Odd as o:
    print("outer", o)
k = kept(7)
print(str(k))
try:
    get(9)
except Odd:
    print("no name")
