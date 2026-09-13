# sorted, sort, min and max with key= and reverse=; stability
words = ["banana", "Apple", "cherry", "date"]
print(sorted(words))
print(sorted(words, key=len))
print(sorted(words, key=lambda w: w.lower()))
print(sorted(words, reverse=True))
print(sorted(words, key=len, reverse=True))
nums = [3, -1, 2, -5]
print(sorted(nums, key=abs))
print(sorted(nums, key=lambda n: -n))
print(sorted(nums, reverse=True))
pairs = [(2, "b"), (1, "z"), (2, "a")]
print(sorted(pairs))
print(sorted(pairs, key=lambda p: p[0]))
print(sorted(pairs, key=lambda p: p[0], reverse=True))
print(sorted(pairs, key=lambda p: p[1]))
xs = [5, 2, 9]
xs.sort(reverse=True)
print(xs)
xs.sort(key=lambda x: x % 5)
print(xs)
print(max(words, key=len), min(words, key=len))
print(max(nums, key=abs), min(nums, key=abs))
people = [{"name": "A", "age": 30}, {"name": "B", "age": 25}]
print(sorted(people, key=lambda p: p["age"])[0]["name"])
print(min(people, key=lambda p: p["age"])["name"])
print(max(people, key=lambda p: p["age"])["name"])
floats = [2.5, -1.0, 0.0]
print(sorted(floats, key=lambda f: -f))
print(sorted("hello"))
print(sorted({"b": 1, "a": 2}))
print(sorted({"b": 1, "a": 2}.items()))
print(sorted({3, 1, 2}))
print(sorted([3, 1, 2], key=str))
