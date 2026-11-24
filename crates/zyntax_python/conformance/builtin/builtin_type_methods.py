# A method of a builtin type called through the type is the method called on its first argument.

words = ["delta", "Alpha", "charlie", "Bravo"]
print(str.upper("abc"), str.lower("ABC"), str.strip("  x  "), str.split("a b c"))
print(str.join(", ", words), str.replace("aXb", "X", "-"), str.startswith("hello", "he"))
print(sorted(words, key=str.lower), list(map(str.upper, words)))
print(str.format("{}-{}", 1, 2), str.count("banana", "a"), str.find("banana", "n"))

xs = [3, 1, 2]
list.append(xs, 4)
list.sort(xs)
print(xs, list.index(xs, 3), list.count(xs, 1))
list.extend(xs, [9, 8])
list.reverse(xs)
last = list.pop(xs)
print(xs, last)

d = {"a": 1}
dict.update(d, b=2)
print(d, dict.get(d, "a"), dict.get(d, "z", 0), sorted(dict.keys(d)), dict.pop(d, "a"))

s = {1, 2}
set.add(s, 3)
print(sorted(s), set.union(s, {9}) == {1, 2, 3, 9})

print(bytes.upper(b"ab"), bytes.fromhex("6162"), bytes.decode(b"hi"))
print(tuple.count((1, 1, 2), 1), tuple.index((5, 6), 6))
