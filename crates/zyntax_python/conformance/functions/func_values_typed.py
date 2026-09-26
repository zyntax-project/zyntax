# Module functions named as values: passed to a higher-order function,
# held in locals, lists and dicts, compared, and called with arguments
# that do not fit.


def square(x):
    return x * x


def halve(x):
    return x / 2


def pair_sum(i_u):
    i, u = i_u
    total = 0
    for j, v in enumerate(u):
        total += (i + j) * v
    return total


def apply_all(func, xs):
    return [func(x) for x in xs]


def apply_pairs(func, u):
    return [func((i, u)) for i in range(len(u))]


def twice(func, x):
    return func(func(x))


def pick(flag):
    if flag:
        return square
    return halve


print(apply_all(square, [1, 2, 3]))
print(apply_all(halve, [1, 2, 3]))
print(apply_all(square, [1.5, 2.5]))
print(apply_pairs(pair_sum, [1, 2, 3]))
print(apply_pairs(pair_sum, [0.5, 1.5]))
print(twice(square, 3), twice(halve, 10))

# A function held in a local, then called.
f = square
print(f(7))
g = pick(False)
print(g(9), pick(True)(4))

# Stored in a list and a dict, and read back.
table = [square, halve]
print([h(6) for h in table])
by_name = {"sq": square, "half": halve}
print(by_name["sq"](5), by_name["half"](5))
print(square == halve)


# Called with arguments the function does not take.
def one(a):
    return a


def call_with_two(func):
    return func(1, 2)


try:
    call_with_two(one)
except TypeError:
    print("TypeError")


# A default filled in through the value.
def scaled(x, k=3):
    return x * k


def call_one(func, x):
    return func(x)


print(call_one(scaled, 2), call_one(scaled, "ab"))


# A generator function as a value.
def count_up(n):
    for i in range(n):
        yield i


def drain(make, n):
    return list(make(n))


print(drain(count_up, 4))


# A module function bound again at module level: every later call,
# from anywhere, reaches the new value.
def first(x):
    return x + 1


def second(x):
    return x * 2


def via_first():
    return first(3)


first = second
print(first(3), via_first())


def fact(n):
    if n <= 1:
        return 1
    return n * fact(n - 1)


def call_fact(n):
    return fact(n)


print(call_fact(5))
old_fact = fact
fact = second
print(call_fact(5), old_fact(4))


def twin():
    return "first twin"


def twin():
    return "second twin"


print(twin())


def rebind_later():
    global target
    target = square


def target(x):
    return -x


def use_target():
    return target(5)


print(use_target())
rebind_later()
print(use_target())
