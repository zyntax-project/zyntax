# A number that holds a float cannot index or count.


def index_with(flag):
    i = 1
    if flag:
        i = 1.0
    try:
        print(list(range(i)))
    except TypeError as e:
        print("TypeError", e)
    try:
        print(list(range(0, 3, i)))
    except TypeError as e:
        print("TypeError", e)


index_with(0)
index_with(1)
