def static_store():
    d3 = {1: 2}
    d3[[1]] = 5
    return d3
try:
    print(static_store())
except TypeError as e:
    print("TypeError", e)
def literal():
    return {[1]: 2}
try:
    print(literal())
except TypeError as e:
    print("TypeError", e)
