# calls with keyword arguments
def greet(name: str, times: int = 1, sep: str = "-") -> str:
    return sep.join([name] * times)

print(greet("ab"))
print(greet("ab", 3))
print(greet("ab", times=2))
print(greet(name="xy", sep="+", times=3))
