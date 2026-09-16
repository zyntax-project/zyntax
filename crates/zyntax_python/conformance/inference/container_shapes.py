def show(pairs):
    for left, right in pairs:
        print(left + right)


pairs = [(1.0, 2.0)]
show(pairs)

def show_alias(pairs):
    for left, right in pairs:
        print(left + right)


another = [(1.0, 2.0)]
same = another
same[0] = (3, 4)
show_alias(another)

def mutate(pairs):
    pairs[0] = (1, 2)


def show_after(pairs):
    for left, right in pairs:
        print(left + right)


third = [(1.0, 2.0)]
mutate(third)
show_after(third)
pairs[0] = (1, 2)
show(pairs)

alias = pairs
alias[0] = (3.0, 4.0)
show(pairs)
