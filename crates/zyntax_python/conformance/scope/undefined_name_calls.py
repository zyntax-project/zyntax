try:
    print(list(map(nosuch, [1, 2])))
except NameError as e:
    print("map", e)

try:
    print(list(filter(nosuch, [1, 2])))
except NameError as e:
    print("filter", e)

try:
    nosuch(print("argument"))
except NameError as e:
    print("call", e)

# A builtin the frontend does not compile is still a name that is bound.
alias = vars
other = getattr
print("bound")
