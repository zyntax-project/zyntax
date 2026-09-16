items = [1, 2, 3]
alias = items
del items[:]
print(items, alias)
items.append(4)
print(items, alias)
