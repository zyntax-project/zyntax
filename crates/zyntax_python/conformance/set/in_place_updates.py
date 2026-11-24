# The in-place set methods change the set every holder sees.

a = {1, 2, 3, 4, 5}
alias = a
a.intersection_update({2, 3, 4, 9})
print(sorted(a), sorted(alias))

b = {1, 2, 3, 4, 5}
b.difference_update({2, 4})
print(sorted(b))

c = {1, 2, 3}
c.symmetric_difference_update({3, 4, 5})
print(sorted(c))

d = {1, 2, 3, 4, 5, 6}
d.intersection_update({2, 3, 4, 5}, {3, 4, 5, 6}, [4, 5, 7])
print(sorted(d))

e = {1, 2, 3, 4}
e.difference_update([1], (2,), {3})
print(sorted(e))

words = {"apple", "banana", "cherry"}
words.difference_update({"banana"})
words.symmetric_difference_update(["cherry", "damson"])
print(sorted(words))

same = {1, 2, 3}
same.intersection_update(same)
print(sorted(same))
same.difference_update(same)
print(same)
