# A set is not JSON; a dict still is, after deletions.
import json
d = {"a": 1, "b": [1, 2], "c": None}
del d["b"]
print(json.dumps(d))
try:
    json.dumps({1, 2})
except TypeError as e:
    print("TypeError", e)
try:
    json.dumps({"s": {3}})
except TypeError as e:
    print("TypeError", e)
