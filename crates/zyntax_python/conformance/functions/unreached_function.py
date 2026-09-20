# a function the program never reaches may use what this frontend lacks
def interactive():
    text = raw_input('?').strip()
    return exec(text)

def helper(x):
    return x * 2

def unused_caller():
    return interactive() + helper(1)

print(helper(21))
