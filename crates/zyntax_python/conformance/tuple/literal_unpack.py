def record(events, value):
    events.append(value)
    return value


events = []
left, right = (record(events, 1), record(events, 2))
print(left, right, events)

first, (second, third) = (record(events, 3), (4.5, "five"))
print(first, second, third)

def get_first(value):
    return value[0]

def last(value):
    return value[-1]

def get_second(value):
    return value[1]


def dict_item(value):
    return value[2]


print(last((10, 20)))
print(get_second([10.5, 20.5]))
print(get_first("abc"))
print(dict_item({2: "two"}))

def dynamic_zero(value):
    return value[0]


print(dynamic_zero((7, 8)))
print(dynamic_zero([7.5, 8.5]))
print(dynamic_zero("xyz"))
print(dynamic_zero({0: "zero"}))

def replace_zero(value):
    value[0] = 9.5
    return value


print(replace_zero([1.5, 2.5]))
print(replace_zero({0: 1.5}))
