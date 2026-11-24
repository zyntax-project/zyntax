import json
import os
import tempfile
import time
from json import loads

data = json.loads('{"name": "zy", "tags": ["a", "b"], "n": 3, "f": 2.5, "ok": true, "none": null, "nested": {"x": [1, {"y": -7}]}}')
print(data["name"], data["tags"], data["n"] + 1, data["f"] * 2, data["ok"], data["none"], data["nested"]["x"][1]["y"])
print(sorted(data), len(data), type(data["n"]).__name__, type(data["f"]).__name__)
print(json.loads("[]"), json.loads("{}"), json.loads(" [1 , 2 ,3 ] "), json.loads("12"), json.loads('"s"'), json.loads("true"), json.loads("null"))
print(json.loads('"line\\nbreak \\u0041 \\"q\\" \\\\ tab\\t"'))
print(json.loads("-0.5e2"), json.loads("1E3"), json.loads("0"), json.loads("[1.0, 2]"))
print(loads('{"a": {"b": {"c": [true, false]}}}')["a"]["b"]["c"])
for bad in ['{"a": }', "[1, 2", '{"a" 1}', "tru", "[1] x"]:
    try:
        json.loads(bad)
    except ValueError as e:
        print("ValueError")
text = json.dumps({"k": [1, 2, {"z": None}], "s": "x"})
print(text, json.loads(text) == {"k": [1, 2, {"z": None}], "s": "x"})
path = os.path.join(tempfile.gettempdir(), "zy_json_" + str(int(time.time() * 1000000)) + ".json")
with open(path, "w") as f:
    f.write('{"items": [10, 20, 30]}')
with open(path) as f:
    loaded = json.load(f)
print(sum(loaded["items"]))
os.remove(path)
