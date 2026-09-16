E = 2
NE = 3
e = 4

def read():
    return E + e

def pick(value=E):
    return value

def lookup(key, values={E: NE}):
    return values[key]

print(read(), pick(), lookup(2))
